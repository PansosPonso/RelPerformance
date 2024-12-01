import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pymc as pm
from nflows import transforms, distributions, flows
import distrax
import sklearn.covariance
import jax.numpy as jnp
import jax
from sklearn.isotonic import IsotonicRegression
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from copulas.multivariate import GaussianMultivariate
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from typing import Union

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")

import random
import lightgbm as lgb
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Embedding, Concatenate, IntegerLookup, Lambda, Activation, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU found.")
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
else:
    print("No GPU found.")

import tensorflow_probability as tfp
from tensorflow.keras.utils import to_categorical

import torch
torch.set_float32_matmul_precision('high')
from gluonts.dataset.pandas import PandasDataset
from lag_llama.lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.model.deepar import DeepAREstimator

import logging
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
import pytorch_lightning as pl
pl.seed_everything(seed=RANDOM_SEED, workers=True)

import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


###########################################################################
######################### Helper Functions: Begin #########################
###########################################################################
# A simple MLP
def create_mlp(
    num_columns, num_hidden_units, hidden_unit, dropout_rate, label_smoothing, learning_rate, investment_id_lookup_layer, investment_id_size
    ):

    investment_id_inputs = Input((1, ), dtype=tf.uint16)
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = Embedding(investment_id_size, 32, input_length=1)(investment_id_x)
    investment_id_x = Reshape((-1, ))(investment_id_x)
    investment_id_x = Dense(32, activation='swish')(investment_id_x)
    investment_id_x = Dense(16, activation='swish')(investment_id_x)

    inp = Input((num_columns, ), dtype=tf.float16)
    x = BatchNormalization()(inp)
    x = Dropout(dropout_rate)(x)
    for i in range(num_hidden_units):
        x = Dense(hidden_unit)(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.swish)(x)
        x = Dropout(dropout_rate)(x)

    x = Concatenate(axis=1)([investment_id_x, x])
    x = Dense(32, activation='swish', kernel_regularizer="l2")(x)

    out = Dense(5, activation='softmax')(x)

    model = Model(inputs=[investment_id_inputs, inp], outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )

    return model

def predict_mlp(model, test_ds, batch_size=256):
    return model.predict(test_ds, batch_size=batch_size)

def preprocess(X, y):
    return X, y

def preprocess_test(investment_id, feature):
    return (investment_id, feature), 0

def make_dataset(feature, investment_id, y, batch_size=1024, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature), y))
    ds = ds.map(preprocess)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def make_test_dataset(feature, investment_id, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature)))
    ds = ds.map(preprocess_test)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

# Code from https://gist.github.com/SaremS/8e5f7256375551e01a9881265c76d161
class MGARCH_DCC(tf.keras.Model):
    """
    Tensorflow/Keras implementation of multivariate GARCH under dynamic conditional correlation (DCC) specification.
    Further reading:
        - Engle, Robert. "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models."
        - Bollerslev, Tim. "Modeling the Coherence in Short-Run Nominal Exchange Rates: A Multi-variate Generalized ARCH Model."
        - Lütkepohl, Helmut. "New introduction to multiple time series analysis."
    """

    def __init__(self, y):
        """
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        super().__init__()
        n_dims = y.shape[1]
        self.n_dims = n_dims

        self.MU = tf.Variable(np.mean(y,0)) #use a mean variable

        self.sigma0 = tf.Variable(np.std(y,0)) #initial standard deviations at t=0

        #we initialize all restricted parameters to lie inside the desired range
        #by keeping the learning rate low, this should result in admissible results
        #for more complex models, this might not suffice
        self.alpha0 = tf.Variable(np.std(y,0))
        self.alpha = tf.Variable(tf.zeros(shape=(n_dims,))+0.25)
        self.beta = tf.Variable(tf.zeros(shape=(n_dims,))+0.25)

        self.L0 = tf.Variable(np.float32(np.linalg.cholesky(np.corrcoef(y.T)))) #decomposition of A_0
        self.A = tf.Variable(tf.zeros(shape=(1,))+0.9)
        self.B = tf.Variable(tf.zeros(shape=(1,))+0.05)


    def call(self, y):
        """
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        return self.get_conditional_dists(y)


    def get_log_probs(self, y):
        """
        Calculate log probabilities for a given matrix of time-series observations
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        return self.get_conditional_dists(y).log_prob(y)


    @tf.function
    def get_conditional_dists(self, y):
        """
        Calculate conditional distributions for given observations
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        T = tf.shape(y)[0]

        #create containers for looping
        mus = tf.TensorArray(tf.float32, size = T) #observation mean container
        Sigmas = tf.TensorArray(tf.float32, size = T) #observation covariance container

        sigmas = tf.TensorArray(tf.float32, size = T+1)
        us = tf.TensorArray(tf.float32, size = T+1)
        Qs = tf.TensorArray(tf.float32, size = T+1)


        #initialize respective values for t=0
        sigmas = sigmas.write(0, self.sigma0)
        A0 = tf.transpose(self.L0)@self.L0
        Qs = Qs.write(0, A0) #set initial unnormalized correlation equal to mean matrix
        us = us.write(0, tf.zeros(shape=(self.n_dims,))) #initial observations equal to zero


        #convenience
        sigma0 = self.sigma0
        alpha0 = self.alpha0**2 #ensure positivity
        alpha = self.alpha
        beta = self.beta
        A = self.A
        B = self.B


        for t in tf.range(T):
            #tm1 = 't minus 1'
            #suppress conditioning on past in notation

            #1) calculate conditional standard deviations
            u_tm1 = us.read(t)
            sigma_tm1 = sigmas.read(t)

            sigma_t = (alpha0 + alpha*sigma_tm1**2 + beta*u_tm1**2)**0.5

            #2) calculate conditional correlations
            u_tm1_standardized = u_tm1/sigma_tm1

            Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims,1))@tf.reshape(u_tm1_standardized, (1,self.n_dims))
            Q_tm1 = Qs.read(t)
            Q_t = A0 + A*(Q_tm1 - A0) + B*(Psi_tilde_tm1 - A0)
            R_t = self.cov_to_corr(Q_t)

            #3) calculate conditional covariance
            D_t = tf.linalg.LinearOperatorDiag(sigma_t)
            Sigma_t = D_t@R_t@D_t


            #4) store values for next iteration
            sigmas = sigmas.write(t+1, sigma_t)
            us = us.write(t+1, y[t,:]-self.MU) #we want to model the zero-mean disturbances
            Qs = Qs.write(t+1, Q_t)

            mus = mus.write(t, self.MU)
            Sigmas = Sigmas.write(t, Sigma_t)


        return tfp.distributions.MultivariateNormalTriL(loc=mus.stack(), scale_tril=tf.linalg.cholesky(Sigmas.stack()))
        #return tfp.distributions.MultivariateNormalFullCovariance(mus.stack(), Sigmas.stack())


    def cov_to_corr(self, S):
        """
        Transforms covariance matrix to a correlation matrix via matrix operations
        Args:
            S: Symmetric, positive semidefinite covariance matrix (tf.Tensor)
        """
        D = tf.linalg.LinearOperatorDiag(1/(tf.linalg.diag_part(S)**0.5))
        return D@S@D


    def train_step(self, data):
        """
        Custom training step to handle keras model.fit given that there is no input-output structure in our model
        Args:
            S: Symmetric, positive semidefinite covariance matrix (tf.Tensor)
        """
        x,y = data
        with tf.GradientTape() as tape:
            loss = -tf.math.reduce_mean(self.get_log_probs(y))

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"Current loss": loss}


    def sample_forecast(self, y, T_forecast = 30, n_samples=500):
        """
        Create forecast samples to use for monte-carlo simulation of quantities of interest about the forecast (e.g. mean, var, corr, etc.)
        WARNING: This is not optimized very much and can take some time to run, probably due to Python's slow loops - can likely be improved
        Args:
            y: numpy.array of training data, used to initialize the forecast values
            T_forecast: number of periods to predict (integer)
            n_samples: Number of samples to draw (integer)
        """
        T = tf.shape(y)[0]

        #create lists for looping; no gradients, thus no tf.TensorArrays needed
        #can initialize directly
        mus = []
        Sigmas = []
        us = [tf.zeros(shape=(self.n_dims,))]
        sigmas = [self.sigma0]
        Qs = []

        #initialize remaining values for t=0
        A0 = tf.transpose(self.L0)@self.L0
        Qs.append(A0)


        #convenience
        sigma0 = self.sigma0
        alpha0 = self.alpha0**2 #ensure positivity
        alpha = self.alpha
        beta = self.beta
        A = self.A
        B = self.B

        #'warmup' to initialize latest lagged features
        for t in range(T):
            #tm1 = 't minus 1'
            #suppress conditioning on past in notation
            u_tm1 = us[-1]
            sigma_tm1 = sigmas[-1]

            sigma_t = (alpha0 + alpha*sigma_tm1**2 + beta*u_tm1**2)**0.5

            u_tm1_standardized = u_tm1/sigma_tm1

            Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims,1))@tf.reshape(u_tm1_standardized, (1,self.n_dims))
            Q_tm1 = Qs[-1]
            Q_t = A0 + A*(Q_tm1 - A0) + B*(Psi_tilde_tm1 - A0)
            R_t = self.cov_to_corr(Q_t)

            D_t = tf.linalg.LinearOperatorDiag(sigma_t)
            Sigma_t = D_t@R_t@D_t


            sigmas.append(sigma_t)
            us.append(y[t,:]-self.MU) #we want to model the zero-mean disturbances
            Qs.append(Q_t)

            mus.append(self.MU)
            Sigmas.append(Sigma_t)


        #sample containers
        y_samples = []
        R_samples = []
        sigma_samples = []


        for n in range(n_samples):

            mus_samp = []
            Sigmas_samp = []
            sigmas_samp = [sigmas[-1]]
            us_samp = [us[-1]]
            Qs_samp = [Qs[-1]]

            #forecast containers
            ys_samp = []
            sig_samp = []
            R_samp = []


            for t in range(T_forecast):
                u_tm1 = us_samp[-1]
                sigma_tm1 = sigmas_samp[-1]
                sigma_t = (alpha0 + alpha**2 + beta*u_tm1**2)**0.5
                u_tm1_standardized = u_tm1/sigma_tm1

                Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims,1))@tf.reshape(u_tm1_standardized, (1,self.n_dims))
                Q_tm1 = Qs_samp[-1]
                Q_t = A0 + A*(Q_tm1 - A0) + B*(Psi_tilde_tm1 - A0)
                R_t = self.cov_to_corr(Q_t)
                D_t = tf.linalg.LinearOperatorDiag(sigma_t)
                Sigma_t = D_t@R_t@D_t
                sigmas_samp.append(sigma_t)
                Qs_samp.append(Q_t)

                ynext = tfp.distributions.MultivariateNormalFullCovariance(self.MU, Sigma_t).sample()
                ys_samp.append(tf.reshape(ynext,(1,1,-1)))
                sig_samp.append(tf.reshape(sigma_t,(1,1,-1)))
                R_samp.append(tf.reshape(R_t,(1,1,self.n_dims,self.n_dims)))

                us_samp.append(ynext-self.MU)

            y_samples.append(tf.concat(ys_samp,1))
            R_samples.append(tf.concat(R_samp,1))
            sigma_samples.append(tf.concat(sig_samp,1))

        return tf.concat(y_samples,0).numpy(), tf.concat(R_samples,0).numpy(), tf.concat(sigma_samples,0).numpy()

# The base class for all models
class Base_Model:
    def __init__(self):
        pass
    def fit(self, train):
        pass
    def forecast(self, test):
        pass

# Code from https://github.com/microprediction/precise/tree/main
X_TYPE = Union[list,np.ndarray,int]
X_DATA_TYPE = Union[list,np.ndarray]
def infer_dimension(n_dim:int=None, x:X_TYPE=None, **ignore)->int:
    """ Infer the number of variables
    :param n_dim:
    :param x:
    :return:
    """
    if n_dim is not None:
        return n_dim
    elif isinstance(x, int):
        return x
    elif len(x)>1:
        return len(x)
    else:
        raise ValueError('Ambiguity in number of variables. Try supplying x or n_dim')

# Code from https://github.com/microprediction/precise/tree/main
def is_data(x):
    return isinstance(x,np.ndarray) or isinstance(x,list)

# Code from https://github.com/microprediction/precise/tree/main
def emp_pcov(s:dict, x:[float]=None, n_dim=None, k=1)->dict:
    """
        Track empirical sample covariance
    """
    assert k==1
    if not s:
        s = _emp_pcov_init(x=x,n_dim=n_dim)
    if is_data(x):
        s = _emp_pcov_update(s=s, x=x)
    return s

# Code from https://github.com/microprediction/precise/tree/main
def _emp_pcov_init(s:dict=None, x:X_DATA_TYPE=None, n_dim=None):
    """ Empirical population covariance"""
    n_dim = infer_dimension(x=x,n_dim=n_dim)
    if s is None:
       s = dict()
    s['n_dim']=n_dim
    s['shape']=(n_dim, n_dim)
    s['ones'] = np.ones(n_dim)
    s['n_samples'] = 0
    s['mean'] = np.zeros(n_dim)
    s['pcov'] = np.eye(n_dim)
    return s

# Code from https://github.com/microprediction/precise/tree/main
def _emp_pcov_update(s:dict, x:X_DATA_TYPE, target=None):
    assert s['n_dim'] == len(x)
    prev_cov = np.copy( s['pcov'] )
    prev_mean = s['mean']
    s['n_samples'] += 1
    s['mean'] = prev_mean + (x - prev_mean)/s['n_samples']
    if target is None:
        delta_x_prev = np.atleast_2d(x-prev_mean)
        delta_x_current = np.atleast_2d(x - s['mean'])
    else:
        delta_x_prev = np.atleast_2d(x - target)
        delta_x_current = np.atleast_2d(x - target)
    s['pcov'] = prev_cov + ( np.matmul( delta_x_current.transpose(),delta_x_prev) - prev_cov ) / s['n_samples']

    return s

# Code from https://github.com/microprediction/precise/tree/main
def run_emp_pcov_d0(y, s:dict, a=None, t=None, e=1):
    """
        Empirical covariance skater that assumes y's are iid
    :param y:  (n_dim,)   Incoming vector of observations
    :param s:             State
    :param a:             Variables know in advance (ignored)
    :param t:             Epoch time (ignored)
    :param e:             Allocated computation time (ignored)
    :return:  (n_dim,), (n_dim,n_dim), dict
              Point estimate, cov estimate, posterior state
    """
    s = emp_pcov(s=s,x=y)
    x = s['mean']
    x_cov = s['pcov']
    return x, x_cov, s

# Calculate the quintiles for each day
def scores_to_quintiles(x):
    sc = np.zeros((x.shape[0], x.shape[1]), dtype=int)
    for i in range(x.shape[0]):
        sc[i,:] = pd.qcut(x.iloc[i], 5, duplicates='drop', labels=False)
    return sc

def isotonic_calibration(in_sample_preds, y_true, preds):
    calibrated_probs = np.zeros_like(preds)

    for j in range(len(np.unique(y_true))):
        # Get the predicted probabilities for class j
        prob_class_j = in_sample_preds[:, j]
        pred_class_j = preds[:, j]

        # Create binary labels: 1 if true label is class j, otherwise 0
        binary_labels = (y_true == j).astype(int)

        # Apply Isotonic Regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')

        # Fit isotonic regression to the predicted probabilities and true binary labels
        iso_reg.fit(prob_class_j, binary_labels)
        calibrated_prob_class_j = iso_reg.transform(pred_class_j)

        # Store the calibrated probabilities for class j
        calibrated_probs[:, j] = calibrated_prob_class_j

    return calibrated_probs / calibrated_probs.sum(axis=1).reshape(-1,1)

###########################################################################
########################## Helper Functions: End ##########################
###########################################################################

###########################################################################
############################### MODELS ####################################
###########################################################################

#################################### Random Forest ####################################
class rforest_model(Base_Model):
    def __init__(self, params=None, feats=None, tickers=None):
        super().__init__()
        self.model = None
        self.params = params
        self.feats = feats
        self.tickers = tickers

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # For speed we use monthly frequency (using daily frequency slows things down without any increase in performance)
        monthly_train_df = train_df[train_df['Date'].isin(pd.date_range(end=max(train_df['Date'].unique()), freq='4W-FRI', periods=train_df['Date'].nunique() / 20 + 1))]
        self.model = RandomForestClassifier(random_state=random.seed(RANDOM_SEED),
                                            criterion=self.params['criterion'],
                                            max_features=self.params['max_features'],
                                            n_estimators=self.params['n_estimators'],
                                            max_depth=self.params['max_depth'],
                                            min_samples_split=self.params['min_samples_split'],
                                            min_samples_leaf=self.params['min_samples_leaf']).fit(monthly_train_df[self.feats], monthly_train_df['target'])

    def forecast(self, test_df):

        pred = self.model.predict_proba(test_df[self.feats].fillna(test_df[self.feats]))

        results = pd.DataFrame(columns = ["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results.ID = self.tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = pred
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### EWMA ####################################
class ewma_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # Instead of caclulating the proportion of times that each asset's return belongs to the 1, 2, 3, 4 and 5 quintile over all
        # previous observations, we only use the last #days observations
        y = scores_to_quintiles(train_df)

        temp = pd.DataFrame(y, columns=train_df.columns, index=train_df.index)
        q_1 = temp.rolling(self.params['days']).agg(lambda x: sum(x == 0) / self.params['days'])
        q_2 = temp.rolling(self.params['days']).agg(lambda x: sum(x == 1) / self.params['days'])
        q_3 = temp.rolling(self.params['days']).agg(lambda x: sum(x == 2) / self.params['days'])
        q_4 = temp.rolling(self.params['days']).agg(lambda x: sum(x == 3) / self.params['days'])
        q_5 = temp.rolling(self.params['days']).agg(lambda x: sum(x == 4) / self.params['days'])
        self.all_q = q_1.reset_index().melt(id_vars=['Date'], value_vars=train_df.columns)
        self.all_q.rename(columns={'value':'Rank1'}, inplace=True)
        train_df_temp = q_2.reset_index().melt(id_vars=['Date'], value_vars=train_df.columns)
        self.all_q['Rank2'] = train_df_temp['value']
        train_df_temp = q_3.reset_index().melt(id_vars=['Date'], value_vars=train_df.columns)
        self.all_q['Rank3'] = train_df_temp['value']
        train_df_temp = q_4.reset_index().melt(id_vars=['Date'], value_vars=train_df.columns)
        self.all_q['Rank4'] = train_df_temp['value']
        train_df_temp = q_5.reset_index().melt(id_vars=['Date'], value_vars=train_df.columns)
        self.all_q['Rank5'] = train_df_temp['value']

    def forecast(self, test_df):
        results = self.all_q.groupby('variable')[['Rank1','Rank2','Rank3','Rank4','Rank5']].ewm(alpha=0.05).mean().reset_index().groupby('variable').last().drop(columns='level_1').reset_index()
        results.rename(columns={'variable':'ID'}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Softmax (or Logistic) Regression ####################################
class logistic_model(Base_Model):
    def __init__(self, params=None, feats=None, tickers=None):
        super().__init__()
        self.model = None
        self.params = params
        self.feats = feats
        self.tickers = tickers

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # For speed we use monthly frequency (using daily frequency slows things down without any increase in performance)
        monthly_train_df = train_df[train_df['Date'].isin(pd.date_range(end=max(train_df['Date'].unique()), freq='4W-FRI', periods=train_df['Date'].nunique() / 20 + 1))]
        self.model = LogisticRegression(random_state=RANDOM_SEED, C=self.params['C']).fit(monthly_train_df[self.feats], monthly_train_df['target'])

    def forecast(self, test_df):

        pred = self.model.predict_proba(test_df[self.feats].fillna(test_df[self.feats]))

        results = pd.DataFrame(columns = ["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results.ID = self.tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = pred
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### SVM ####################################
class svm_model(Base_Model):

    def __init__(self, params=None, feats=None, tickers=None):
        super().__init__()
        self.model = None
        self.params = params
        self.feats = feats
        self.tickers = tickers

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # For speed we use monthly frequency (using daily frequency slows things down without any increase in performance)
        monthly_train_df = train_df[train_df['Date'].isin(pd.date_range(end=max(train_df['Date'].unique()), freq='4W-FRI', periods=train_df['Date'].nunique() / 20 + 1))]
        self.model = svm.SVC(probability=True, kernel=self.params['kernel'], random_state=RANDOM_SEED, C=self.params['C']).fit(monthly_train_df[self.feats], monthly_train_df['target'])

    def forecast(self, test_df):
        pred = self.model.predict_proba(test_df[self.feats].fillna(test_df[self.feats]))

        results = pd.DataFrame(columns = ["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results.ID = self.tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = pred
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Nauve Bayes ####################################
class naive_bayes_model(Base_Model):
    def __init__(self, params=None, feats=None, tickers=None):
        super().__init__()
        self.model = None
        self.params = params
        self.feats = feats
        self.tickers = tickers

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # For speed we use monthly frequency (using daily frequency slows things down without any increase in performance)
        monthly_train_df = train_df[train_df['Date'].isin(pd.date_range(end=max(train_df['Date'].unique()), freq='4W-FRI', periods=train_df['Date'].nunique() / 20 + 1))]
        self.model = GaussianNB(var_smoothing=self.params['var_smoothing']).fit(monthly_train_df[self.feats], monthly_train_df['target'])

    def forecast(self, test_df):
        pred = self.model.predict_proba(test_df[self.feats].fillna(test_df[self.feats]))

        results = pd.DataFrame(columns = ["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results.ID = self.tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = pred
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Normalizing Flows ####################################
class flows_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # Define an invertible transformation
        transform = transforms.CompositeTransform([
            transforms.MaskedAffineAutoregressiveTransform(features=train_df.shape[1], hidden_features=self.params['hidden_features'],
                                                           num_blocks=self.params['num_blocks'], dropout_probability=self.params['dropout_probability']),
        ])

        # Define a base distribution
        base_distribution = distributions.StandardNormal(shape=[train_df.shape[1]])
        self.model = flows.Flow(transform=transform, distribution=base_distribution)

        optimizer = torch.optim.Adam(self.model.parameters())
        num_iter = self.params['iter']
        for i in range(num_iter):
            x = torch.tensor(train_df.values, dtype=torch.float32)
            optimizer.zero_grad()
            loss = -self.model.log_prob(inputs=x).mean()
            loss.backward()
            optimizer.step()

    def forecast(self, test_df):
        synthetic_data = self.model.sample(1000)
        synthetic_data = pd.DataFrame(data=synthetic_data.detach().numpy(), columns=test_df.columns)

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Multivariate Normal Distribution with Garch ####################################
class garch_mnd_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        s = {}
        for y in train_df.values:
            x_mean, x_cov, s = run_emp_pcov_d0(s=s, y=y)
        cov = x_cov

        # Use a Dynamic conditional correlation multivariate GARCH model to forecast the covariance matrix
        mgarch_train = train_df.copy()
        self.model = MGARCH_DCC(mgarch_train)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['lr']))
        self.model.fit(mgarch_train, mgarch_train, batch_size=mgarch_train.shape[0], shuffle=False, epochs=self.params['epochs'], verbose=False)

        conditional_dists = self.model(mgarch_train.values)
        if np.isnan(conditional_dists.covariance()[-1,:,:].numpy()).sum().sum() == 0:
            cov = conditional_dists.covariance()[-1,:,:]

        # Generate data using a Multivariate Normal Distribution with zero means and the Garch forecasted covariance matrix
        self.model = distrax.MultivariateNormalFullCovariance(jnp.array(np.zeros((train_df.shape[1],))), jnp.array(cov))

    def forecast(self, test_df):

        synthetic_data = self.model.sample(seed=jax.random.PRNGKey(RANDOM_SEED), sample_shape=[1000])
        synthetic_data = pd.DataFrame(data=synthetic_data.__array__(), columns=test_df.columns)

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Kernel Density ####################################
class kde_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        self.model = KernelDensity(bandwidth=self.params['bandwidth']).fit(train_df)

    def forecast(self, test_df):
        synthetic_data = self.model.sample(n_samples=1000, random_state=RANDOM_SEED)

        synthetic_data = pd.DataFrame(data=synthetic_data, columns=test_df.columns)

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Gaussian Mixture ####################################
class gaussian_mixture_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        self.model = GaussianMixture(n_components=self.params['n_components'], random_state=RANDOM_SEED).fit(train_df)

    def forecast(self, test_df):
        synthetic_data = self.model.sample(n_samples=1000)[0]
        synthetic_data = pd.DataFrame(data=synthetic_data, columns=test_df.columns)

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### VAE ####################################
class vae_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # We first standardize data and then use monthly frequency for speed purposes
        scaled_train_df = pd.DataFrame(data=(train_df.values - train_df.mean(axis=1).values.reshape((train_df.shape[0],1))) / train_df.std(axis=1).values.reshape((train_df.shape[0],1)), columns=train_df.columns, index=train_df.index)
        monthly_return_df = scaled_train_df[scaled_train_df.index.isin(pd.date_range(end=scaled_train_df.index[-1], freq='4W-FRI', periods=scaled_train_df.shape[0] / 20 + 1))]

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(monthly_return_df)

        self.model = TVAESynthesizer(metadata, epochs=self.params['epochs'], batch_size=self.params['batch_size'],
                                    compress_dims=self.params['compress_dims'], decompress_dims=self.params['compress_dims'])
        self.model.fit(monthly_return_df)

    def forecast(self, test_df):
        synthetic_data = self.model.sample(num_rows=1000)

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### LLama ####################################
class llama_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # We first standardize data and then use monthly frequency for speed purposes
        scaled_train_df = pd.DataFrame(data=(train_df.values - train_df.mean(axis=1).values.reshape((train_df.shape[0],1))) / train_df.std(axis=1).values.reshape((train_df.shape[0],1)), columns=train_df.columns, index=train_df.index)
        self.dataset = PandasDataset(dict(scaled_train_df))

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            ckpt = torch.load("lag_llama/lag-llama.ckpt", map_location=torch.device('cuda:0'))
        else:
            ckpt = torch.load("lag_llama/lag-llama.ckpt", map_location=torch.device('cpu'))

        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
            ckpt_path="lag_llama/lag-llama.ckpt",
            prediction_length=1,
            context_length=self.params["context_length"],
            lr=self.params["lr"],
            input_size=estimator_args["input_size"],
            n_layer=self.params["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=True,
            batch_size=self.params['batch_size'],
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger":False,
                "log_every_n_steps":200,
                "max_epochs": self.params['max_epochs'],
            },
        )

        self.predictor = estimator.train(self.dataset, cache_data=True, shuffle_buffer_length=1000)

    def forecast(self, test_df):

        forecast_it = self.predictor.predict(self.dataset, num_samples=1000)

        synthetic_data = list(forecast_it)
        synthetic_data = pd.DataFrame(data=[[item[0] for item in synthetic_data[i].samples] for i in range(test_df.shape[1])], index=test_df.columns).T

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### DeepAR ####################################
class deepar_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        scaled_train_df = pd.DataFrame(data=(train_df.values - train_df.mean(axis=1).values.reshape((train_df.shape[0],1))) / train_df.std(axis=1).values.reshape((train_df.shape[0],1)), columns=train_df.columns, index=train_df.index)
        self.dataset = PandasDataset(dict(scaled_train_df))

        estimator = DeepAREstimator(
            freq='D',
            prediction_length=1,
            hidden_size=self.params['hidden_size'],
            num_layers=self.params['num_layers'],
            context_length=self.params['context_length'],
            dropout_rate=self.params['dropout_rate'],
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger":False,
                "log_every_n_steps":200,
                "max_epochs": self.params['max_epochs'],
            },
        )

        self.predictor = estimator.train(self.dataset, cache_data=True)


    def forecast(self, test_df):

        forecast_it = self.predictor.predict(self.dataset, num_samples=1000)

        synthetic_data = list(forecast_it)
        synthetic_data = pd.DataFrame(data=[[item[0] for item in synthetic_data[i].samples] for i in range(test_df.shape[1])], index=test_df.columns).T

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### PatchTST ####################################
class patchtst_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        scaled_train_df = pd.DataFrame(data=(train_df.values - train_df.mean(axis=1).values.reshape((train_df.shape[0],1))) / train_df.std(axis=1).values.reshape((train_df.shape[0],1)), columns=train_df.columns, index=train_df.index)
        self.dataset = PandasDataset(dict(scaled_train_df))

        estimator = PatchTSTEstimator(
            prediction_length=1,
            dim_feedforward=self.params['dim_feedforward'],
            patch_len=self.params['patch_len'],
            nhead=self.params['nhead'],
            d_model=self.params['d_model_multiplier'] * self.params['nhead'],
            dropout=self.params['dropout'],
            lr=self.params['lr'],
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger":False,
                "log_every_n_steps":200,
                "max_epochs": self.params['max_epochs'],
            },
        )

        self.predictor = estimator.train(self.dataset, cache_data=True)


    def forecast(self, test_df):

        forecast_it = self.predictor.predict(self.dataset, num_samples=1000)

        synthetic_data = list(forecast_it)
        synthetic_data = pd.DataFrame(data=[[item[0] for item in synthetic_data[i].to_sample_forecast(num_samples=1000).samples] for i in range(test_df.shape[1])], index=test_df.columns).T

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### MLP ####################################
class mlp_model(Base_Model):
    def __init__(self, params=None, feats=None, tickers=None):
        super().__init__()
        self.model = None
        self.params = params
        self.feats = feats
        self.tickers = tickers

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # Encode the asset ids, since they will do not change
        investment_id = train_df['Stock']
        investment_ids = list(investment_id.unique())
        investment_id_size = len(investment_ids) + 1
        investment_id_lookup_layer = IntegerLookup(max_tokens=investment_id_size)
        investment_id_lookup_layer.adapt(pd.DataFrame({"investment_ids":investment_ids}))

        tf.keras.backend.clear_session()

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(train_df[self.feats])
        X = pd.DataFrame(X).fillna(0.5)
        train_ds = make_dataset(X, train_df['Stock'], to_categorical(train_df['target']))

        self.model = create_mlp(len(self.feats), self.params['num_hidden_units'], self.params['hidden_unit'], self.params['dropout_rate'],
                         self.params['label_smoothing'], self.params['learning_rate'], investment_id_lookup_layer, investment_id_size)

        self.model.fit(train_ds, epochs=self.params['epochs'], batch_size=256, verbose=0)

    def forecast(self, test_df):
        X = self.scaler.transform(test_df[self.feats])
        X = pd.DataFrame(X).fillna(0.5)
        test_ds = make_test_dataset(X, test_df['Stock'])

        pred = predict_mlp(self.model, test_ds, batch_size=256)

        results = pd.DataFrame(columns = ["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results.ID = self.tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = pred
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### MVN ####################################
class mnd_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        # We use the robust covariance method to caclulate the covariance matrix
        robust_cov = sklearn.covariance.MinCovDet(assume_centered=True, random_state=RANDOM_SEED).fit(train_df)
        cov = robust_cov.covariance_

        self.model = distrax.MultivariateNormalFullCovariance(jnp.array(np.zeros((train_df.shape[1],))), jnp.array(cov))

    def forecast(self, test_df):

        synthetic_data = self.model.sample(seed=jax.random.PRNGKey(RANDOM_SEED), sample_shape=[1000])
        synthetic_data = pd.DataFrame(data=synthetic_data.__array__(), columns=test_df.columns)

        y = scores_to_quintiles(synthetic_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### Gaussian Copula ####################################
class copula_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        self.model = GaussianMultivariate()
        self.model.set_random_state(RANDOM_SEED)
        self.model.fit(train_df)

    def forecast(self, test_df):

        sampled = self.model.sample(1000)

        y = scores_to_quintiles(sampled)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### LGBM ####################################
class lgbm_model(Base_Model):
    def __init__(self, params=None, feats=None, tickers=None):
        super().__init__()
        self.model = None
        self.params = params
        self.feats = feats
        self.tickers = tickers

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        fixed_params = {
                        'boosting_type': 'gbdt',
                        'objective': 'multiclass',
                        'metric': 'multi_logloss',
                        'num_class':5,
                        'linear_tree':True,
                        'verbose':-1,
                        'random_state':RANDOM_SEED
                        }

        # For speed we use monthly frequency (using daily frequency slows things down without any increase in performance)
        monthly_train_df = train_df[train_df['Date'].isin(pd.date_range(end=max(train_df['Date'].unique()), freq='4W-FRI', periods=train_df['Date'].nunique() / 20 + 1))]
        train_set = lgb.Dataset(monthly_train_df[self.feats], monthly_train_df['target'], free_raw_data=False, categorical_feature=['AssetClass'])
        self.model = lgb.train(dict(fixed_params, **self.params), train_set, num_boost_round=50, categorical_feature=['AssetClass'])

        self.in_sample_forecast = pd.DataFrame(self.model.predict(monthly_train_df[self.feats]))
        self.y_train = pd.DataFrame(monthly_train_df['target'])
        self.groups = pd.DataFrame(monthly_train_df['Date'])

    def forecast(self, test_df):

        pred_lgb = self.model.predict(test_df[self.feats])

        '''
        from sklearn.model_selection import GroupKFold
        calibrated_preds = np.zeros_like(pred_lgb)
        group_kfold = GroupKFold(n_splits=5)
        for i, (train_index, test_index) in enumerate(group_kfold.split(self.in_sample_forecast, self.y_train, self.groups)):
            calibrated_preds += isotonic_calibration(self.in_sample_forecast.iloc[train_index].values, self.y_train.iloc[train_index].values.reshape(-1,), pred_lgb)

        calibrated_preds /= 5

        calibrated_preds /= calibrated_preds.sum(axis=1).reshape(-1,1)
        '''

        results = pd.DataFrame(columns = ["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results.ID = self.tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = pred_lgb
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### GAN ####################################
class gan_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        metadata = SingleTableMetadata()
        scaled_train_df = pd.DataFrame(data=(train_df.values - train_df.mean(axis=1).values.reshape((train_df.shape[0],1))) / train_df.std(axis=1).values.reshape((train_df.shape[0],1)), columns=train_df.columns, index=train_df.index)
        metadata.detect_from_dataframe(scaled_train_df)

        self.model = CTGANSynthesizer(metadata, epochs=self.params['epochs'], batch_size=self.params['batch_size'],
                                    generator_dim=self.params['generator_dim'], discriminator_dim=self.params['generator_dim'])
        self.model.fit(train_df)
        self.train_df = train_df

    def forecast(self, test_df):

        gan_data = self.model.sample(num_rows=1000)

        y = scores_to_quintiles(gan_data)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results

#################################### BVAR ####################################
# Code from: https://www.pymc-labs.com/blog-posts/bayesian-vector-autoregression/
class bvar_model(Base_Model):
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params

    def fit(self, train_df, hyperparams=None):

        if hyperparams is not None:
            self.params = hyperparams

        coords={
            "lags": reversed(range(-self.params['lags'], 0)),
            "vars": tuple(train_df.columns),
            "cross_vars": tuple(train_df.columns),
            "time": range(len(train_df) - self.params['lags']),
        }

        with pm.Model(coords=coords) as BVAR_model:
            # Priors for the model intercept, lagged coefficients and noise terms
            intercept = pm.Normal("intercept", mu=0, sigma=1, dims=("vars",))
            lag_coefs = pm.Normal("lag_coefs", mu=0, sigma=1, dims=("lags", "vars", "cross_vars"))
            noise = pm.HalfNormal("noise", dims=("vars",))

            # Compute autoregressive expectation for each variable
            ar_ = []
            for j in range(0,len(coords['vars'])):
                ar_.append(pm.math.sum([
                    pm.math.sum(lag_coefs[i, j] * train_df.values[self.params['lags']-(i+1): -(i+1)], axis=-1)
                    for i in range(self.params['lags'])
                ], axis=0))

            # Stack autoregressive terms and add intercepts
            mean = intercept + pm.math.stack([ar_[j] for j in range(0,len(coords['vars']))], axis=-1)

            obs = pm.Normal("obs", mu=mean, sigma=noise, observed=train_df[self.params['lags']:], dims=("time", "vars"))

        with BVAR_model:
            trace = pm.sample(chains=1, tune=10, random_seed=rng, progressbar=False, nuts_sampler='blackjax')

        self.post = trace.posterior.stack(sample=("chain", "draw"))

    def forecast(self, test_df):

        # Function that takes a single draw of parameters and forecasts n steps
        def _forecast(intercept, lag_coefs, noise, forecast=1):
            new_draws = np.zeros((forecast, test_df.shape[1]))
            for i in range(forecast):
                ar_ = []
                for j in range(0,test_df.shape[1]):
                    ar_.append(np.sum(lag_coefs[:, j] * new_draws[i-self.params['lags']: i]))
                mean = intercept + np.stack([ar_[j] for j in range(0,test_df.shape[1])])
                new_draws[i] = rng.normal(mean, noise)
            return new_draws

        # Vectorized forecast function to handle multiple parameter draws
        bvar_forecast = np.vectorize(
            _forecast,
            signature=("(v),(l,v,v),(v)->(o,v)"),
            excluded=("forecast",),
        )

        # Take a random subset of 1000 draws from the posterior
        draws = rng.integers(1000, size=1000)

        intercept_draws = self.post["intercept"].values.T[draws]

        lag_coefs_draws = self.post["lag_coefs"].values.T[draws].T
        lag_coefs_draws = np.moveaxis(lag_coefs_draws, -1, 0)

        noise_draws = self.post["noise"].values.T[draws]

        # Forecast 1 month into the future
        ppc_draws = bvar_forecast(intercept_draws, lag_coefs_draws, noise_draws, forecast=1)
        ppc_draws = np.swapaxes(ppc_draws, 0, 1)

        bvar_forecasts = pd.DataFrame(data=ppc_draws[-1,:,:], columns=test_df.columns)

        y = scores_to_quintiles(bvar_forecasts)
        p = list()
        for i in range(5):
            pi = np.mean(y==i,axis=0)
            p.append(pi)

        results = pd.DataFrame(columns=test_df.columns, data=p).transpose()
        results = results.reset_index()
        results.rename(columns={'index':'ID',0:"Rank1",1:"Rank2",2:"Rank3",3:"Rank4",4:"Rank5"}, inplace=True)
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += 0.2 - results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].mean()

        return results


torch.cuda.empty_cache()
jax.clear_caches()
