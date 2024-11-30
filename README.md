# Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches 
Author: Panagiotis Samartzis

## Introduction
This repository contains the code for the paper titled "Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches." The project includes various models to evaluate financial assets' relative performance using different datasets and methodologies.

## Contents
The main contents of the repository are the following:

- data/ folder: This folder contains the raw input data:
  - _Data_M6.xlsx_ and _Data_v2.xlsx_: price & volume data for M6 and M6+ samples,
  - _Categorizations.xlsx_: asset & model categorizations,
  - _F-F_Research_Data_Factors_daily.xlsx_: data from Kenneth Data Library and
  - _Universe_M6.xlsx_ and _Universe_v2.xlsx_: the asset universes for M6 and M6+ samples.
- output/ folder: This folder contains the excel files generated by Calculate_id_vol.py, Generate_forecasts.py and Calculate_metrics.py:
  - _Idiosyncratic_vol.xlsx_: Idiosyncratic volatility for each sample (the output of Calculate_id_vol.py)
  - _Results_M6.xlsx_ and _Results_v2.xlsx_: Models' forecasts for M6 and M6+ samples, respectively (the output of Generate_forecasts.py)
  - _Results_M6_with_metrics.xlsx_ and _Results_v2_with_metrics.xlsx_: Models' forecasts together with evaluation metrics for for M6 and M6+ samples, respectively (the output of Calculate_metrics.py)
- results/ folder: This folder contains the HTML files for all tables and JPG files for all figures in the paper (the outputs of Calculate_tables.py)
- lag_llama/ folder: The code for the Lag-Llama model which has been obtained from https://github.com/time-series-foundation-models/lag-llama.

> [!NOTE]
> You can run all scripts for custom data, assuming they have the same structure as the ones for the M6 or M6+ samples.

> [!WARNING]
> To run _Generate_forecasts.py_ and _Calculate_id_vol.py_ (see below for more information), make sure Data_M6.xlsx and Data_v2.xlsx (or any other custom Data_other.xlsx) are filled with data from a provider. These files are sample templates provided to illustrate the required structure.

## How to use this repository
If you haven’t installed Git, download it from https://git-scm.com/ and follow the instructions for your operating system.

Clone this repository:
```
git clone https://github.com/PansosPonso/RelPerformance.git
```

Navigate to the project folder:
```
cd RelPerformance
```

Set up a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # For Windows, use: source venv/Scripts/activate
```

Install dependencies for Ubuntu or WSL2 (to run on GPU, check this guide: https://pradeepl.com/blog/installing-nvidia-cuda-tensorflow-on-windows-wsl2):
```
pip install -r requirements_Ubuntu.txt
```

Install dependencies for Windows (TensorFlow does not support GPU on Windows):
```
pip install -r requirements_Windows.txt 
```

Run the relevant python scripts (see below).


## Usage to replicate the results of the paper
The following code generates the 9 tables and 2 figures for the paper "Predicting the Relative Performance among Financial Assets: A Comparative Analysis of Different Approaches.". 

To calculate all evaluation metrics for both M6 and M6+ samples (or other custom data), use the following command:

```
python Calculate_metrics.py --REPLICATE_PAPER 1
```
or, more generally,
```
python Calculate_metrics.py --FILE_NAME 'outputs/Results_M6.xlsx' --TUNING_SAMPLE 12
```
***Arguments***

- **REPLICATE_PAPER**: If set to 1, calculates all evaluation metrics for both M6 and M6+ samples.
- **TUNING_SAMPLE**: The number of observations in the tuning sample (12 for the M6 sample and 36 for the M6+ sample).
- **FILE_NAME**: Specifies the file that contains the forecasts.

> [!NOTE]
> This code takes ~1 hour for the M6 sample and ~6 hours for the M6+ sample, using a laptop with an AMD Ryzen 5 5600H CPU and Nvidia GeForce RTX 3050 GPU.


To generate Tables 2-10 and Figures 1-2, use the following command **(after running Calculate_metrics.py)**:

```
python Calculate_tables.py --REPLICATE_PAPER 1
```
or, more generally,
```
python Calculate_tables.py --FILE_NAME 'outputs/Results_M6.xlsx' --TUNING_SAMPLE 12 --FREQ 6
```

***Arguments***

- **REPLICATE_PAPER**: If set to 1, calculates Tables and Figures of the paper.
- **TUNING_SAMPLE**: The number of observations in the tuning sample.
- **FREQ**: The frequency (in months) that the ensemble models are reevaluated.
- **FILE_NAME**: Specifies the file that contains the forecasts and the metrics.

> [!NOTE]
> This code takes <1 min for both M6 and M6+ samples, using a laptop with an AMD Ryzen 5 5600H CPU and Nvidia GeForce RTX 3050 GPU.

## Usage to generate new forecasts
To run the models and generate the forecasts for the M6 sample (or other custom data), use the following command:

```
python Generate_forecasts.py --MODEL_NAMES MND MLP LGBM RF SR DeepAR PatchTST KDE GM GC NF VAE NB LagLlama SVM GAN BVAR --SAMPLE 'M6'
```

***Arguments***

- **SAMPLE**: 'M6' for M6 sample, 'M6+' for M6+ sample and 'other' for other custom data.
- **KEEP_TUNING_FORECASTS**: Specifies whether to keep the forecasts that were generated for the tuning sample (it is used for calibrating forecasts).
- **MODEL_NAMES**: Specifies the models to be evaluated. Available choices are listed below.
- **DATA_FROM_PREVIOUS_RUN**: In case the code was interrupted, this parameter allows specifying an Excel file containing the forecasts generated up to the point of interruption.
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
> Note that the results of most models can vary between runs due to inherent randomness that cannot be fully controlled. Additionally, results may vary depending on whether the models are trained on a CPU or GPU.

> [!NOTE]
> This code takes ~3 hours for the M6 sample and ~21 hours for the M6+ sample, using a laptop with an AMD Ryzen 5 5600H CPU and Nvidia GeForce RTX 3050 GPU. When running on CPU, the time is approximately 4 times longer.

## Idiosyncratic volatility calculation
To calculate idiosyncratic volatility per asset for the M6 and M6+ samples (or other custom data), use the following command:

```
python Calculate_id_vol.py --SAMPLE 'M6' --FILE_NAME 'outputs/Results_M6.xlsx' --DATA_FILE 'data/Data_M6.xlsx'
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
