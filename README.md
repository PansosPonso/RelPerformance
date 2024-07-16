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

- **M6**: If set to 1, evaluates models using the M6 data and evaluation period.
- **YAHOO**: If set to 1, obtains data from Yahoo Finance.
- **ENSEMBLE**: If set to 1, creates the ensemble model.
- **TOP_N_MODELS**: Specifies the number of models the ensemble model will consist of.
- **MODEL_NAMES**: Specifies the models to be evaluated. Available choices are listed below.
  
***Available Models***

- **RF**: Random Forest
[Random Forests by Breiman (2001)] (https://link.springer.com/article/10.1023/A:1010933404324)
- **SR**: Softmax (or multinomial logistic) Regression
- **SVM**: Support Vector Machine
[Support-Vector Networks by Cortes and Vapnik (1995)](https://link.springer.com/article/10.1007/BF00994018)
- **NB**: Naive Bayes
- **MLP**: Multilayer Perceptron
[Multilayer Feedforward Networks Are Universal Approximators by Hornik, Stinchcombe, and White (1989)] (https://www.nature.com/articles/323533a0)
- **LGBM**: LightGBM (Light Gradient Boosting Machine)
[LightGBM: A Highly Efficient Gradient Boosting Decision Tree by Ke et al. (2017)] (https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- **KDE**: Kernel Density Estimator
- **GM**: Gaussian Mixture
[Finite Mixture Models by McLachlan and Peel (2000)] (https://www.wiley.com/en-us/Finite+Mixture+Models-p-9780471006268)
- **NF**: Normalizing Flows
[Variational Inference with Normalizing Flows by Rezende and Mohamed (2015)] (https://arxiv.org/abs/1505.05770)
- **VAE**: Variational Autoencoder
[Auto-Encoding Variational Bayes by Kingma and Welling (2013)] (https://arxiv.org/abs/1312.6114)
- **LagLlama**: Lag Llama
[Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting by Rasul et al. (2023)] (https://arxiv.org/abs/2310.08278)
- **DeepAR**: DeepAR
[DeepAR: Probabilistic forecasting with autoregressive recurrent networks by Salinas et al. (2023)] (https://arxiv.org/abs/1704.04110)
- **PatchTST**: PatchTST
[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers by Nie et al. (2023)] (https://arxiv.org/abs/2211.14730)
- **MND**: Multivariate Normal Distribution with robust covariance matrix
- **MND+GARCH**: Multivariate Normal Distribution with DCC GARCH covariance matrix
- **GC**: Gaussian Copula
- **GAN**: Generative Adversarial Network
[Generative Adversarial Nets by Goodfellow et al. (2014)] (https://arxiv.org/abs/1406.2661)
- **BVAR**: Bayesian Vector Auto-Regression model
[Forecasting and conditional projection using realistic prior distributions by Doan, Litterman and Sims (1984)] (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=305579)

## Contributing
We welcome contributions to improve the project. Please feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
