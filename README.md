![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) &emsp;
![VSCODE](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white) &emsp;
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white) &emsp;
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

<br />

Table of Contents
=================

* [Computational Finance Coursework](#computational-finance-coursework)
   * [Tasks](#tasks)
      * [Part I - ARIMA Model](#part-i---arima-model)
      * [Part II - Algorithmic Trading](#part-ii---algorithmic-trading)
      * [Part III - Portfolio Optimisation](#part-iii---portfolio-optimisation)
   * [Final Report](#final-report)
      * [Contributors](#contributors)

# Computational Finance Coursework

This report presents the implementation methodology and
findings of investigations into three computational finance
concepts; time series analysis, algorithmic trading and port-
folio optimisation. For each part, the implementation and
testing approaches are described and justified, and the results
analysed.

## Tasks
### Part I - ARIMA Model
An ARIMA(2, 1, 1) model is implemented from scratch and is initially used for some simple forecasting. 

Subsequently, daily prices are obtained from Microsoft
(MSFT), Apple (APPL) and Tesco (TSCO) on the FTSE 100 index and are used to train and tune the hyperparameters of the ARIMA model via gradient descent. 

The model is subsequently evaluated on a test set using the mean absolute percentage error (MAPE).

<p align="center">
  <img src=READMEimgs/ARIMA.png width="900"/>
</p>

### Part II - Algorithmic Trading
A pairs trading approach is applied to a suitable pair of stocks from the FTSE 100 index, identified by using the Pearson Correlation coefficient. 

As a result,  Tesco (TSCO) and Pershing Square Holdings Ltd (PSH) were selected for Pairs trading.

Two pairs trading strategies are then implemented and compared using the obtained returns on the test set as a metric for strategy performance.

<p align="center">
  <img src=READMEimgs/PairsTradingResults.png width="350"/>
</p>

### Part III - Portfolio Optimisation
5 stocks (AutoTrader, Experian, Rightmove, Rolls Royce and Shell) are selected from the FTSE 100 index and split into training and test sets. 

The returns and covariances of the stocks are estimated based on the training data.

An efficient portfolio is derived from the efficient frontier, that we derive from a grid search performed over all defined combinations of the weight vector. The resulting efficient portfolio is then compared to a baseline $\frac 1 n$ portfolio on the testset.

<p align="center">
  <img src=READMEimgs/EfficientFrontier.png width=500/>
</p>

## Final Report
The results and analysis are discussed in the [final report](report.pdf).

### Contributors
`Charlie Powell`

`Benjamin Sanati`

`Oran Bramble`