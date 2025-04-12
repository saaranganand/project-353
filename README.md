<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]

<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Stock Index Prediction Using Machine Learning</h1>
      <br />
      <a href="https://docs.google.com/document/d/1o63-8nKeNCyImbiUbQZqw-AMwAPUFRTUGoyTA65ZLfE/edit?usp=sharing">Detailed Report</a>
    </summary>
  </ul>
</div>

</div>

<!-- ABOUT THE PROJECT -->
## About The Project
Our goal is to predict the future performance of a stock index (we chose VFV.TO - Vanguard S&P 500 Index ETF), using price data from 2019 to Present Day. The project involves data collection & preprocessing, feature engineering, model training, and evaluation based on real-world financial data.

We aim to use machine learning techniques to predict price movements for this index. Specifically, we will 3 train models (1 each) using data from 2019 to 2025 and generate predictions for January-December 2025, which we will validate using January-March 2025 data. Additionally, we aim to forecast the future direction of the index (until the end of the year).


## Steps Involved:

- Data Collection & Processing
- Data Analysis
- Model Training and Validation
- Results Visualization


<!-- GETTING STARTED -->
## Getting Started

To get the code up and running, follow these steps:

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/saaranganand/project-353.git
   ```
2. Navigate to the repo
   ```sh
   cd project-353
   ```
3. Install prerequisite packages
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Run <code>data_collect.py</code>
   ```sh
   python data_collect.py
   ```
   - This generates <code>data.csv</code>, which is the dataset that will be used by all 3 of our machine learning models.

2. Run <code>VFV.TO_plots.py</code>
   ```sh
   python VFV.TO_plots.py
   ```
   - This generates plots with general information about <code>data.csv</code>
     - Daily Returns
     - Stock Price, with both 50 day and 200 day moving averages
     - Volatility

---

### 3. <ins>KNN</ins>

Running these files will produce plots/figures in the `/plots/` directory

To run the KNN classifier, simply type

```sh
python KNN_analysis/knn_classification_forecast.py
```

You should see that a file called `confusion_matrix.png` has been created

To run the KNN regressor, type

```sh
python KNN_analysis/knn_regression_forecast.py
```

This will generate 2 plots: `knn_validation_plus_forcast.png` and `knn_validation.png`

---

### 4. <ins>Neural Net (MLP Regressor)</ins>

**a.** `mlp_regressor_preprocess.py`

- Do not run this script, it does not have a main function.
  - Sole purpose is to serve as a helper to <code>mlp_regressor_train.py</code>
- This code accepts a dataset (in this case, <code>data.csv</code>)
- It loads and preprocess raw stock index data for MLP regression.

**b.** Run `mlp_regressor_train.py`

```sh
python neuralnet/mlp_regressor_train.py
```
- This code accepts a dataset (in this case, <code>data.csv</code>)
- It loads and preprocess raw stock index data for MLP regression (using <code>mlp_regressor_preprocess.py</code>).
- It generates 2 plots under the <code>/neuralnet/plots</code> directory
  - <code>mlp_validation_plot.png</code>: Actual vs. predicted prices for the validation period
  - <code>price_forecast_full.png</code>: Stock index price forecasting for the rest of the year

- Terminal output should look something like this:
```
Training the MLPRegressor model...
Validation MAE: 1.96
Validation RMSE: 2.67
Forecasted price on 2025-12-31: 99.07 CAD

Process finished with exit code 0
```
---

### 5. <ins>LSTM (Long Short-Term Memory) Model </ins>

### a. Running the `LSTM_Stock_Predictor.py` Script

1. **Open your command line (terminal)** and navigate to the directory `..\project-353\LSTM_Model` containing the `LSTM_Stock_Predictor.py` file.

2. **Run the script** with the following command:
   ```bash
   python LSTM_Stock_Predictor.py


The Terminal will give an output similar to this
```
" Training the model...
Epoch 1/50 "
```
This indicates the model is being run and you will have to wait a couple minuites before the model is finished excuting.

The terminal will then output 
```
"Forecasted price on 2025-12-31: $xx.xx CAD which is the prediction for VFV.TO at the end of the year.
```
This output is the prediction for VFV.TO at the end of the year.
An output file called **LSTM_Future_Forcast.png** will been created in the same directory.
This graph is a visual interpretation predicting the growth of the ETF throughout the year. 

```
---
---
## Overall Results/Findings:

---

### The Team:

<a href="https://github.com/saaranganand/project-353/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=saaranganand/project-353" alt="contrib.rocks image" />
</a>

* [Amardeep Sangha](https://github.com/Amar710/)
* [James Chuong](https://github.com/JamesChuong)
* [Saarang Anand](https://github.com/saaranganand/)
  
---

_CMPT 353 - Group AJS_\
_Simon Fraser University - Spring 2025_



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/saaranganand/project-353.svg?style=for-the-badge
[contributors-url]: https://github.com/saaranganand/project-353/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/saaranganand/project-353.svg?style=for-the-badge
[stars-url]: https://github.com/saaranganand/project-353/stargazers
