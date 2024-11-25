import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calc_pauc(y_test, y_hat, fpr=0.2):
    pAUC = 0
    for i in range(5):
        pAUC += roc_auc_score((y_test[:,i] > 0) * 1, y_hat[:,i], max_fpr=fpr)

    return pAUC / 5


def bootstrapped_CI_auc(y_true, y_pred):
    n_bootstraps = 1000
    bootstrapped_scores = []

    rng = np.random.RandomState(RANDOM_SEED)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))

        try:
            score = calc_pauc(y_true[indices,:], y_pred[indices,:], fpr=0.25)
            bootstrapped_scores.append(score)
        except:
            pass

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def bootstrapped_CI_acc(y_true, y_pred):
    n_bootstraps = 1000
    bootstrapped_scores = []

    rng = np.random.RandomState(RANDOM_SEED)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))

        score = sum(y_pred[indices] == y_true[indices]) / len(y_pred)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def calc_ece(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def isotonic_calibration(in_sample_preds, in_sample_y_true, out_sample_preds):
    calibrated_probs = np.zeros_like(out_sample_preds)

    for j in range(len(np.unique(in_sample_y_true))):
        # Get the predicted probabilities for class j
        prob_class_j = in_sample_preds[:, j]
        pred_class_j = out_sample_preds[:, j]

        # Create binary labels: 1 if true label is class j, otherwise 0
        binary_labels = (in_sample_y_true == j).astype(int)

        # Apply Isotonic Regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')

        # Fit isotonic regression to the predicted probabilities and true binary labels
        iso_reg.fit(prob_class_j, binary_labels)
        calibrated_prob_class_j = iso_reg.transform(pred_class_j)

        # Store the calibrated probabilities for class j
        calibrated_probs[:, j] = calibrated_prob_class_j

    return calibrated_probs / calibrated_probs.sum(axis=1).reshape(-1,1)


def caclulate_metrics(file_name, back):

    m_names = ['KDE','GM','MND','GC','NF','DeepAR','GAN','VAE','Lag-Llama','NB','SR','LGBM','RF','SVM','MLP','BVAR','PatchTST','EWMA']

    forecasts_data = pd.read_excel(file_name, sheet_name='forecasts')
    tickers = forecasts_data.iloc[2:102,1].values.tolist()
    forecasts = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([(d, asset) for d in forecasts_data.iloc[2:,0][~pd.isna(forecasts_data.iloc[2:,0])].values for asset in tickers], names=['Date', 'Asset']),
                columns=pd.MultiIndex.from_tuples([(m, r) for m in [name for name in forecasts_data.columns if "Unnamed" not in name][1:len(m_names)+1]+['truth'] for r in ["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]], names=['Models', 'Ranks']),
                dtype=float
        )

    forecasts.iloc[:,:] = pd.concat([forecasts_data.iloc[2:,2:len(m_names) * 5 + 2], forecasts_data.iloc[2:,-5:]], axis=1).values


    info_names = []
    for c in m_names:
        for w in [c + '_RPS', c + '_Acc', c + '_KL', c + '_ECE', c + '_pAUC', c + '_ACCBoot',c + '_PAUCBoot']:
            info_names.append(w)

    information = pd.DataFrame(
                index=forecasts.index.get_level_values(0).unique(),
                columns=info_names,
                dtype=float
            )

    calibrated_information = pd.DataFrame(
                index=forecasts.index.get_level_values(0).unique(),
                columns=info_names,
                dtype=float
            )

    rps_assets = pd.DataFrame(
                index=tickers,
                columns=m_names,
                data=0,
                dtype=float
            )


    calibrated_forecasts = forecasts.copy()
    for c in m_names:
        print('Calibrate ', c)
        for d in forecasts.index.get_level_values(0).unique()[1:]:
            temp = forecasts.loc[forecasts.loc[:, (c, )].index.get_level_values(0) < d, (c, )].iloc[-back * 100:,:].reset_index().drop(columns=['Asset','Date'])
            y_train = forecasts.loc[forecasts.loc[:, ('truth', )].index.get_level_values(0) < d, ('truth', )].iloc[-back * 100:].values.argmax(axis=1)
            calibrated_forecasts.loc[d, (c, )] = isotonic_calibration(temp.values, y_train, forecasts.loc[d, (c, )].values)


    for c in m_names:
        print(c)
        for i,d in enumerate(forecasts.index.get_level_values(0).unique()):

            acc = sum(forecasts.loc[d, (c, )].values.argmax(axis=1) == forecasts.loc[d, ('truth', )].values.argmax(axis=1)) / 100

            acc_CI_low, acc_CI_high = bootstrapped_CI_acc(forecasts.loc[d, (c, )].values.argmax(axis=1), forecasts.loc[d, ('truth', )].values.argmax(axis=1))


            frc = np.cumsum(forecasts.loc[d, (c, )],axis=1)
            target = np.cumsum(forecasts.loc[d, ('truth', )],axis=1)
            rps = np.mean(np.mean(np.power((target - frc), 2), axis=1))

            if i >= back:
                rps_assets.loc[:, c] += np.mean(np.power((target - frc), 2), axis=1)

            kl = np.sum(np.log((forecasts.loc[d, (c, )] / 0.2)) * forecasts.loc[d, (c, )], axis=1).mean()

            ece = calc_ece(forecasts.loc[d, (c, )].values, forecasts.loc[d, ('truth', )].values.argmax(axis=1))

            pauc = calc_pauc(forecasts.loc[d, ('truth', )].values, forecasts.loc[d, (c, )].values, fpr=0.25)
            auc_CI_low, auc_CI_high = bootstrapped_CI_auc(forecasts.loc[d, ('truth', )].values, forecasts.loc[d, (c, )].values)

            information[c + '_RPS'].loc[d] = rps
            information[c + '_Acc'].loc[d] = acc
            information[c + '_KL'].loc[d] = kl
            information[c + '_ECE'].loc[d] = ece
            information[c + '_pAUC'].loc[d] = pauc
            information[c + '_ACCBoot'].loc[d] = acc_CI_low
            information[c + '_PAUCBoot'].loc[d] = auc_CI_low

            ############## The same for calibrated forecasts ###################

            acc = sum(calibrated_forecasts.loc[d, (c, )].values.argmax(axis=1) == calibrated_forecasts.loc[d, ('truth', )].values.argmax(axis=1)) / 100

            acc_CI_low, acc_CI_high = bootstrapped_CI_acc(calibrated_forecasts.loc[d, (c, )].values.argmax(axis=1), calibrated_forecasts.loc[d, ('truth', )].values.argmax(axis=1))


            frc = np.cumsum(calibrated_forecasts.loc[d, (c, )],axis=1)
            target = np.cumsum(calibrated_forecasts.loc[d, ('truth', )],axis=1)
            rps = np.mean(np.mean(np.power((target - frc), 2), axis=1))

            kl = np.sum(np.log((calibrated_forecasts.loc[d, (c, )] / 0.2)) * calibrated_forecasts.loc[d, (c, )], axis=1).mean()

            ece = calc_ece(calibrated_forecasts.loc[d, (c, )].values, calibrated_forecasts.loc[d, ('truth', )].values.argmax(axis=1))

            pauc = calc_pauc(calibrated_forecasts.loc[d, ('truth', )].values, calibrated_forecasts.loc[d, (c, )].values, fpr=0.25)
            auc_CI_low, auc_CI_high = bootstrapped_CI_auc(calibrated_forecasts.loc[d, ('truth', )].values, calibrated_forecasts.loc[d, (c, )].values)

            calibrated_information[c + '_RPS'].loc[d] = rps
            calibrated_information[c + '_Acc'].loc[d] = acc
            calibrated_information[c + '_KL'].loc[d] = kl
            calibrated_information[c + '_ECE'].loc[d] = ece
            calibrated_information[c + '_pAUC'].loc[d] = pauc
            calibrated_information[c + '_ACCBoot'].loc[d] = acc_CI_low
            calibrated_information[c + '_PAUCBoot'].loc[d] = auc_CI_low

        rps_assets.loc[:, c] /= (forecasts.index.get_level_values(0).nunique() - back)


    with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        information.to_excel(writer, sheet_name='Info')
        calibrated_information.to_excel(writer, sheet_name='Infocalib')
        rps_assets.to_excel(writer, sheet_name='RPS_assets')


if __name__ == '__main__':

    # To run: python Calculate_metrics.py --file_name Results_m6.xlsx --tuning_sample 12

    parser = argparse.ArgumentParser(description='Calculate metrics')
    parser.add_argument('--file_name', nargs='+', type=str, help="The file that contains the forecasts")
    parser.add_argument('--tuning_sample', nargs='?', type=int, const=0, default=12)
    args = parser.parse_args()

    caclulate_metrics(args.file_name, args.tuning_sample)


    print('\nTask completed...')
    print('\n###################################################################################')