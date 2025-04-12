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
## Overall Results/Findings:
Robust Predictive Performance:

The implemented machine learning models—KNN, MLP Regressor, and LSTM—effectively captured market trends using historical price data from January 2019 to the present.

The LSTM model achieved a validation RMSE of about 2.5 CAD and forecasted an approximate 8% increase in VFV.TO prices by December 2025.

The MLP Regressor provided strong one-day-ahead predictions with an MAE as low as 1.79 CAD and an RMSE of 2.41 CAD, successfully mapping short-term fluctuations.

Comprehensive Data Processing & Feature Engineering:

Utilized yfinance to gather over 1,200 trading days of data, ensuring data integrity with forward-fill techniques and normalization via MinMaxScaler.

Engineered key features such as daily returns, 50-day and 200-day moving averages, and 30-day rolling volatility to enrich the predictive capability of the models.

Effective Model Architectures & Training Efficiency:

The LSTM model was architected with two layers (128 and 64 units) augmented by a 20% dropout rate, optimizing learning with callbacks (EarlyStopping with a 5-epoch patience and ReduceLROnPlateau with a 50% reduction factor) and converging within 50 epochs at a batch size of 32.

The MLP and KNN models were also fine-tuned via hyperparameter optimization and cross-validation techniques, ensuring a balanced trade-off between bias and variance.

Accurate Validation and Forecasting:

Predictions were validated on data from January to March 2025, with visualizations clearly showing that predicted values closely tracked actual market prices.

Iterative forecasting methods allowed each model to project daily prices until the end of 2025, with the MLP and LSTM models demonstrating smooth trend capture despite minor iterative drift.

Actionable Visual Insights:

Detailed plots overlaying historical training data, validation outcomes, and future forecasts provided transparent, data-driven insights.

Visualizations were crucial for identifying specific market events (e.g., volatility during COVID-19 and the impact of recent tariffs) that influenced model performance.

Identified Limitations & Considerations for Future Work:

KNN: While computationally efficient, KNN is sensitive to extreme values and requires careful hyperparameter tuning to avoid overfitting.

MLP: Although adept at short-term predictions, the iterative forecast exhibited error compounding and a slight downward bias due to simplified volatility assumptions.

LSTM: Despite robust performance in sequential prediction, the LSTM model’s long-term forecasts are susceptible to unpredictability in volatile markets, partly due to the absence of external economic indicators.

Future enhancements could include integrating additional macroeconomic data, employing alternative architectures (such as GRU or Transformer models), and developing methods to better quantify uncertainty.

Collaborative Impact & Performance Improvement:

Team contributions, including specialized efforts in deep learning and time-series analysis, resulted in up to a 15% performance improvement over initial baselines.

Cross-functional collaboration enabled the integration of multiple modeling approaches, offering comprehensive insights into VFV.TO price movements and positioning the project as a replicable framework for further financial analytics.


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
