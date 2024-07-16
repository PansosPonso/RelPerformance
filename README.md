# Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches 

## Introduction
This repository contains the code for the paper titled "Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches." The project includes various models to evaluate financial assets' relative performance using different datasets and methodologies.

## Installation
First, install all dependencies using:

```
pip install -r requirements.txt
```

Usage
To run the code, use the following command:

```
python Evaluate.py --M6 1 --YAHOO 0 --ENSEMBLE 0 --TOP_N_MODELS 4 --MODEL_NAMES VAE GAN
```

Optional Arguments

-**M6**: If set to 1, evaluates models using the M6 data and evaluation period.
-**YAHOO**: If set to 1, obtains data from Yahoo Finance.
**ENSEMBLE**: If set to 1, creates the ensemble model.
**TOP_N_MODELS**: Specifies the number of models the ensemble model will consist of.
**MODEL_NAMES**: Specifies the models to be evaluated. Available choices are listed below.
Available Models
**RF**: Random Forest
Random Forests by Breiman (2001)
**SR**: Softmax (or logistic) Regression
**SVM**: Support Vector Machine
[Support-Vector Networks by Cortes and Vapnik (1995)](https://www.example.com)
**NB**: Naive Bayes
**MLP**: Multilayer Perceptron
Multilayer Feedforward Networks Are Universal Approximators by Hornik, Stinchcombe, and White (1989)
**LGBM**: LightGBM (Light Gradient Boosting Machine)
LightGBM: A Highly Efficient Gradient Boosting Decision Tree by Ke et al. (2017)
**KDE**: Kernel Density Estimator
**GM**: Gaussian Mixture
**NF**: Normalizing Flows
**VAE**: Variational Autoencoder
Auto-Encoding Variational Bayes by Kingma and Welling (2013)
**LagLlama**: Lag Llama
DeepAR: DeepAR
**DeepAR**: Probabilistic Forecasting with Autoregressive Recurrent Networks by Salinas et al. (2017)
**PatchTST**: PatchTST
**Average**: Simple benchmark model 1
**EWMA**: Simple benchmark model 2
**MND**: Multivariate Normal Distribution with robust covariance matrix
**MND+GARCH**: Multivariate Normal Distribution with DCC covariance matrix
**GC**: Gaussian Copula
The Gaussian Copula by Cherubini, Luciano, and Vecchiato (2004)
**GAN**: Generative Adversarial Network
Generative Adversarial Nets by Goodfellow et al. (2014)
**BVAR**: Bayesian Vector Auto-Regression model

## Contributing
We welcome contributions to improve the project. Please feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
We would like to thank the contributors and the community for their valuable feedback and support.
