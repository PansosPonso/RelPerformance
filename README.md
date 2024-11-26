# Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches 

Author: Panagiotis Samartzis

## Introduction
This repository contains the code for the paper titled "Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches." The project includes various models to evaluate financial assets' relative performance using different datasets and methodologies.

## Contents
The main contents of the repository are the following:
- \Data folder:
- \Output folder:


## Installation
First, install all dependencies using:

```
pip install -r requirements.txt
```

## Usage to replicate the results of the paper
The following code generates the 9 tables and 2 figures for the paper "Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches.". 

To calculate all evaluation metrics, use the following command:

```
python Calculate_metrics.py --file_name Results_m6.xlsx --tuning_sample 12
```

***Arguments***

- **tuning_sample**: The number of observations in the tuning sample (12 for the M6 sample and 36 for the M6+ sample).
- **file_name**: Specifies the file that contains the forecasts.

> [!NOTE]
> This code takes ~1 hour for the M6 sample and ~6 hours for the M6+ sample, using a laptop with an AMD Ryzen 5 5600H CPU and Nvidia GeForce RTX 3050 GPU.


To calculate all tables, use the following command **(after running Calculate_metrics.py)**:

```
python Calculate_tables.py --replicate_paper 1
```
OR, more generally,
```
python Calculate_tables.py --file_name 'Results_m6.xlsx' --data_file 'Data_M6.xlsx' --tuning_sample 12 --freq 6
```

***Arguments***

- **replicate_paper**: If set to 1, calculates Tables and Figures of the paper.
- **tuning_sample**: The number of observations in the tuning sample.
- **freq**: The frequency (in months) that the ensemble models are reevaluated.
- **file_name**: Specifies the file that contains the forecasts and the metrics.
- **data_file**: Specifies the file that contains the price data.

> [!NOTE]
> This code takes <1 min for the M6 sample and <1 min for the M6+ sample, using a laptop with an AMD Ryzen 5 5600H CPU and Nvidia GeForce RTX 3050 GPU.

## Usage to generate new forecasts
To run the models and generate the forecasts, use the following command:

```
python Generate_forecasts.py --MODEL_NAMES MND MLP LGBM RF SR DeepAR PatchTST KDE GM GC NF VAE NB LagLlama SVM GAN BVAR --KEEP_TUNING_FORECASTS 1 --M6 1 --data_from_previous_run Results_m6.xlsx
```

***Arguments***

- **M6**: If set to 1, evaluates models using the M6 data and evaluation period.
- **YAHOO**: If set to 1, data are obtained from Yahoo Finance.
- **KEEP_TUNING_FORECASTS**: Specifies whether to keep the forecasts that were generated for the tuning sample.
- **MODEL_NAMES**: Specifies the models to be evaluated. Available choices are listed below.
- **data_from_previous_run**: In case the code was interrupted, this parameter allows specifying an Excel file containing the forecasts generated up to the point of interruption.
- **TUNING**: Indicates whether the models' hyperparameters should be tuned.
  
***Available Models***

- **KDE** (Kernel Density Estimator): [Remarks on Some Nonparametric Estimates of a Density Function by Rosenblatt (1956)](https://www.jstor.org/stable/2237390)
- **GC** (Gaussian Copula): [Fonctions de répartition à n dimensions et leurs marges by sklar (1959)](https://hal.science/hal-04094463/document)
- **NB** (Naive Bayes): [Pattern Classification and Scene Analysis by Duda and Hart (1973)](https://www.semanticscholar.org/paper/Pattern-classification-and-scene-analysis-Duda-Hart/b07ce649d6f6eb636872527104b0209d3edc8188)
- **BVAR** (Bayesian Vector Autoregressive): [Forecasting and conditional projection using realistic prior distributions by Doan, Litterman and Sims (1984)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=305579)
- **SR** (Softmax (or Multinomial) Regression): [Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition by Bridle (1989)](https://link.springer.com/chapter/10.1007/978-3-642-76153-9_28)
- **MLP** (Multilayer Perceptron): [Multilayer Feedforward Networks Are Universal Approximators by Hornik, Stinchcombe, and White (1989)](https://www.nature.com/articles/323533a0)
- **SVM** (Support Vector Machine): [Support-Vector Networks by Cortes and Vapnik (1995)](https://link.springer.com/article/10.1007/BF00994018)
- **GM** (Gaussian Mixture): [Finite Mixture Models by McLachlan and Peel (2000)](https://www.wiley.com/en-us/Finite+Mixture+Models-p-9780471006268)
- **RF** (Random Forests): [Random Forests by Breiman (2001)](https://link.springer.com/article/10.1023/A:1010933404324)
- **MND+GARCH** (Multivariate Normal Distribution with DCC GARCH covariance matrix): [Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models by Engle (2002)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=236998)
- **MND** (Multivariate Normal Distribution with robust covariance matrix): [A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices by Ledoit and Wolf (2004)](https://www.sciencedirect.com/science/article/pii/S0047259X03000964)
- **VAE** (Variational Autoencoder): [Auto-Encoding Variational Bayes by Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114)
- **GAN**: [Generative Adversarial Nets by Goodfellow et al. (2014)](https://arxiv.org/abs/1406.2661)
- **NF** (Normalizing Flows): [Variational Inference with Normalizing Flows by Rezende and Mohamed (2015)](https://arxiv.org/abs/1505.05770)
- **LGBM**: [LightGBM: A Highly Efficient Gradient Boosting Decision Tree by Ke et al. (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- **LagLlama**: [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting by Rasul et al. (2023)](https://arxiv.org/abs/2310.08278)
- **DeepAR**: [DeepAR: Probabilistic forecasting with autoregressive recurrent networks by Salinas et al. (2023)](https://arxiv.org/abs/1704.04110)
- **PatchTST**: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers by Nie et al. (2023)](https://arxiv.org/abs/2211.14730)

> [!IMPORTANT]
> Note that for the following models—DeepAR, PatchTST, NF, VAE, LagLlama, and GAN—results may vary as their internal randomness cannot be fixed.

> [!NOTE]
> This code takes ~3 hours for the M6 sample and ~21 hours for the M6+ sample, using a laptop with an AMD Ryzen 5 5600H CPU and Nvidia GeForce RTX 3050 GPU.


## Contributing
We welcome contributions to improve the project. Please feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
