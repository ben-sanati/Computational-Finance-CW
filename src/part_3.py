import copy
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math

def get_stock_data(stock_path):
    """
    Functions which gets all data for all stocks in the stock path passed in as an argument

    :param stock_path: Directory where stock csv files are stored
    :return: List of lists
    """
    # Gets all file paths from the stock_dir
    all_files = [os.path.join(stock_path, f) for f in os.listdir(stock_path) if os.path.isfile(os.path.join(stock_path, f))]
    all_stocks_data = []

    for file in all_files:
        one_stocks_data = []
        with open(file, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                try:
                    stock_close = float(row[4])
                    one_stocks_data.append(stock_close)
                # For when we pass the first line of one of the files which contains the table header's instead of data
                # so converting one of these strings to a float causes a ValueError
                except ValueError:
                    pass

        all_stocks_data.append(one_stocks_data)

    return all_stocks_data


def split_all_stock_data(stocks_data, ratios):
    """
    Function which splits all stock data into training and testin sets

    :param stocks_data:
    :param ratios:
    :return:
    """
    all_train, all_test = [], []
    for stock, ratio in zip(stocks_data, ratios):
        train, test = split_single_stock_data(stock, ratio)
        all_train.append(train)
        all_test.append(test)
    return all_train, all_test


def split_single_stock_data(stock_data, ratio):
    """
    Function which splits a single stock's data into training and testing sets

    :param stock_data:
    :param ratio:
    :return:
    """
    num_in_train = round(ratio*len(stock_data))
    train, test = [], []
    for s in range(len(stock_data)):
        if s < num_in_train:
            train.append(stock_data[s])
        else:
            test.append(stock_data[s])
    return train, test


def calculate_all_returns(data):
    """
    Function which calculates all returns for all stocks

    :param data:
    :return:
    """
    returns = [calculate_single_returns(stock_data) for stock_data in data]
    expected_returns = [sum(single_stock_returns)/len(single_stock_returns) for single_stock_returns in returns]
    return returns, expected_returns

def calculate_single_returns(data):
    """
    Function which calculates single returns

    :param data:
    :return:
    """
    x = list(zip(data, data[1:]))
    return_f = lambda a: ((a[1] - a[0]) / a[0]) * 100
    returns = list(map(return_f, x))
    return returns


def calculate_covariance(returns):
    """
    Function which calculates the covariance matrix of the stocks

    :param returns: List of lists, with each list being a single stocks returns
    :return: Covariance matrix
    """
    # Adjusts each stock's returns by the average of that stocks return
    adjusted_returns = np.array([np.subtract(one_stock_returns, (sum(one_stock_returns)/len(one_stock_returns)))
                        for one_stock_returns in returns])
    cov_matrix = np.divide(np.matmul(adjusted_returns, adjusted_returns.T), len(returns[0])-1)
    return cov_matrix


def calculate_expected_returns(weight_vector, expected_returns):
    """
    Function which calculates weighted expected returns

    :param weight_vector: List being the weight vector to calculate returns for
    :param expected_returns: List being the expected returns for each stock in the unweighted portfolio
    :return: Float being the expected returns of the weight vector
    """
    return np.matmul(weight_vector, expected_returns)


def find_optimal_weights(train_data, weight_combos, risk_level, cov_matrix):
    """
    Function which finds the weight vector/portfolio with the maximum expected returns

    :param train_data: List of list containing training data
    :param weight_combos: List of lists, each list containing a weight vector
    :param risk_level: Float, value which dictates what risk level we should optimise return for
    :return: List being the optimal weight vector
    """
    optimal_weight_vector = []
    optimal_returns = 0
    _, expected_returns = calculate_all_returns(train_data)
    for weight_vector in weight_combos:
       # print(expected_returns, weight_vector)
        sd = get_standard_devs(cov_matrix, [weight_vector])[0]
        if sd <=  risk_level:
            weighted_returns = np.matmul(weight_vector, expected_returns)
            if weighted_returns > optimal_returns:
                optimal_weight_vector = copy.deepcopy(weight_vector)
                optimal_returns = copy.deepcopy(weighted_returns)
    return optimal_weight_vector


def get_test_expected_returns(weight_vectors, test_data):
    """
    Function which calculates the expected returns on the test set

    :param weight_vectors: List of lists, each list containing a weight vector
    :param test_data: List of lists containing test data
    :return: List, with each element being an expected return corresponding to a weight vector
    """

    _, expected_returns = calculate_all_returns(test_data)
    ers = [np.matmul(w, expected_returns) for w in weight_vectors]
    return ers


def get_standard_devs(cov_matrix, weight_vectors):
    """
    Function that calculates the standard deviation of each portfolio (weight vector)

    :param cov_matrix: List of lists, each list containing row of covariance matrix
    :param weight_vectors: List of lists, each list containing a weight vector
    :return: List, each element being a standard deviation
    """
    sds = []
    for weight_vector in weight_vectors:
        sd = sum([math.sqrt(cov_matrix[i][i])*weight_vector[i] for i in range(len(weight_vector)) if weight_vector[i] != 0])
        sds.append(sd)
    return sds


def get_all_weight_combinations(num_stocks):
    """
    Function that calculates all weight combinations. Done by discretising the search space into 0.02 steps from
    0.0 to 1 (0.0, 0.02, 0.04, ... 0.98, 1.0). Then, finds all combinations of this using the itertools product method
    and then filter these to only select ones which sum to 1

    :param num_stocks: Int, number of stocks in portfolio
    :return: List of lists, each list containing a valid weight vector
    """
    vector_length = num_stocks
    elements_range = [x / 100 for x in range(2, 102, 2)]
    combinations = product(elements_range, repeat=vector_length)
    validate_combinations_1 = [c for c in combinations if sum(c) == 1]
    return validate_combinations_1


def plot_risk_return(weight_vectors, cov_matrix, all_returns, expected_returns, size, title):
    """
    Function to plot the risk vs return graphs

    :param weight_vectors: List of lists, each list being a weight vector
    :param cov_matrix: List of lists, each list being a row of the covariance matrix
    :param all_returns: List, each element being the returns of a corresponding weight vector (in same index)
    :param size: Float, size of points on plots
    :param title: String, either Training or Testing, used for title of plots
    :return:
    """
    risk_free_rate = 0
    all_risk = get_standard_devs(cov_matrix, weight_vectors)
    sharpe_ratios = [(r-risk_free_rate)/sd for r, sd in zip(all_returns, all_risk)]
    plt.scatter(all_risk, all_returns, c=sharpe_ratios, s=size)
    if title == "Training":
        # Plotting optimal portfolio point 0.02, 0.02, 0.02, 0.24, 0.7)
        x_risk = get_standard_devs(cov_matrix, [[0.02, 0.02, 0.02, 0.24, 0.7]])
        x_return = calculate_expected_returns([0.02, 0.02, 0.02, 0.24, 0.7], expected_returns)
        plt.scatter(x_risk, x_return, c="black")
        # Plotting one-over-n
        x_risk = get_standard_devs(cov_matrix, [[0.2,0.2,0.2,0.2,0.2]])
        x_return = calculate_expected_returns([0.2,0.2,0.2,0.2,0.2], expected_returns)
        plt.scatter(x_risk, x_return, c="red")
    plt.colorbar(label="Sharpe Ratio")
    plt.title("Risk vs Return Plot ({})".format(title))
    plt.xlabel("RISK (Standard Deviation)")
    plt.ylabel("EXPECTED RETURNS")
    plt.show()


def print_test_results(returns, weight_vectors, risks):
    """
    Function to output test results to the console

    :param returns: List, each element is the returns of a respective weight vector (in same index)
    :param weight_vectors: List of lists, each list containing a weight vector which corresponds to a certain return
    :param risks: List, each element is the standard deviation of the portfolio with each weight vector
    :return: None
    """
    print("OUTPUTTING RETURNS\n")
    for re, w, ri in zip(returns, weight_vectors, risks):
        print("Weight : {} \nReturns : {} \nSD : {}\n".format(w, re, ri))


def plot_cov_matrix(cov_matrix):
    """
    Function to plot the covariance matrix

    :param cov_matrix:
    :return:
    """
    fig, axes = plt.subplots(1, 1, figsize=(6, 18))
    fig.tight_layout(pad=5.0)
    final_stock_tickers = ["AutoTrader", "Experian", "Rightmove", "RollsRoyce", "Shell"]
    # plotting the correlation matrix
    image = axes.imshow(cov_matrix, cmap='coolwarm')
    fig.colorbar(image, ax=axes, label='Covaraince')
    axes.set_title('FTSE100 Stock Covariance Matrix')
    axes.set_xticks(np.arange(len(cov_matrix)))
    axes.set_xticklabels(final_stock_tickers, rotation=90)
    axes.set_yticks(np.arange(len(cov_matrix)))
    axes.set_yticklabels(final_stock_tickers)
    axes.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    axes.set_aspect('equal')  # Set aspect ratio to equal
    plt.show()

def print_standard_devs(cov_matrix):
    """
    Method that outputs all standard deviations

    :param cov_matrix:
    :return:
    """
    sds = [math.sqrt(cov_matrix[i][i]) for i in range(len(cov_matrix))]
    print(sds)


def plot_returns_vs_time(data, weights):
    """
    Function that plots the returns of portfolios (described by their weight vectors) over time

    :param data:
    :param weights:
    :return:
    """
    returns, _ = calculate_all_returns(data)
    labels = []
    for weight_vector in weights:
        total_returns = []
        first_loop = True
        for stock_returns, stock_weight in zip(returns, weight_vector):
            #print(stock_weight, stock_returns)
            x = np.multiply(stock_returns, stock_weight)
            if first_loop:
                total_returns = x
            else:
                total_returns = np.add(total_returns, x)
        days = [x for x in range(1, len(total_returns)+1)]
        plt.plot(days, total_returns)
        labels.append(weight_vector)
    plt.legend(labels)
    plt.ylabel("RETURNS")
    plt.xlabel("TIME")
    plt.show()


def optimal_weight_run(train, weight_combos, covariance_mat, num_stocks, expected_returns, test):
    """
    Function to run the training to find the optimal wreight vector/efficient portfolio, then runs this on the
    test set, and compares it to a one-over-n vector on the test set

    :param train: List of lists of training data
    :param weight_combos: List of lists, each list containing a weight vector to try
    :param covariance_mat: List of lists, each list being a row of the covariance matrix
    :param num_stocks: Int, number of stocks in portfolio
    :param expected_returns: List, each element being the expected returns of each stock in the portfolio (without being weighted)
    :param test: List of lists of test data
    :return: None
    """
    # Gets a list of all expected returns for every weight vector combination. used to plot risk vs return later
    train_returns = [calculate_expected_returns(w, expected_returns) for w in weight_combos]
    # Initialise the even split weight vector (1/n for each stock)
    one_over_n = [1 / num_stocks for stock in range(num_stocks)]
    one_over_n_risk = get_standard_devs(covariance_mat, [one_over_n])[0]
    # Then find the weight vector/portfolio with maximum expected returns over the stocks
    optimal_weight_vector = find_optimal_weights(train, weight_combos, one_over_n_risk, covariance_mat)

    test_weight_vectors = [optimal_weight_vector, one_over_n]
    # This gets the expected returns on the test set for both the optimal and even split weight vectors
    test_returns = get_test_expected_returns(test_weight_vectors, test)
    # Calculates the risk (standard deviation) of each of these portfolios using the covariance matrix
    test_sds = get_standard_devs(covariance_mat, test_weight_vectors)
    # Outputs the results on the test set
    print_test_results(test_returns, test_weight_vectors, test_sds)
    plot_returns_vs_time(test, test_weight_vectors)
    # Plots the risk vs return for the training set, with every weight vector combination and its expected return

    plot_risk_return(weight_combos, covariance_mat, train_returns, expected_returns, 4, "Training")
    # Plots risk vs return for the test set, so risk vs return for optimal weight vector and one-over-n vector on
    # test set
    plot_risk_return(test_weight_vectors, covariance_mat, test_returns, expected_returns, 40, "Testing")


def part_3():
    # Gets returns information for every stock in directory 'assets/'
    # List of list with each inner list containing a single stock's returns for a year
    stock_data = get_stock_data("assets/")
    # Number of stocks saved
    num_stocks = len(stock_data)
    # Splits the stock_data into train and test lists
    # Both train and test are lists of lists, with each inner list being a single stock's returns for a year
    train_test_ratios = [0.7, 0.7, 0.7, 0.7, 0.7]
    train, test = split_all_stock_data(stock_data, train_test_ratios)
    # Gets the returns and expected returns of these stocks and uses these to calculate the covariance matrix
    returns, expected_returns = calculate_all_returns(train)
    covariance_mat = calculate_covariance(returns)
    plot_cov_matrix(cov_matrix=covariance_mat)
   # print_standard_devs(covariance_mat)
    # Outputs the covariance matrix
    print("PORTFOLIO INFO : ")
    print(expected_returns)
    print(covariance_mat)
    print()
    # This discretesizes the search space, and gets all combinations of weight vector, each representing a portfoli
    weight_combos = get_all_weight_combinations(num_stocks)
    # This runs the training to find the weight vector/portfolio with optimal expected returns, then tests this on
    # the test set, and compares it to a one-over-n vector also on the test set, where all monetary allocation is even.
    optimal_weight_run(train, weight_combos, covariance_mat, num_stocks, expected_returns, test)


if __name__ == "__main__":
    part_3()