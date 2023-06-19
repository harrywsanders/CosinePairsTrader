import pandas as pd
import numpy as np
import yfinance as yf
import random
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.tsa.arima_model import ARIMA
from multiprocessing import Pool
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_pairs_return(stock1: str, stock2: str, data: pd.DataFrame) -> float:
    try:
        stock1_data = data[stock1]
        stock2_data = data[stock2]

        stock1_pct_change = (stock1_data['Close'][-1] - stock1_data['Close'][-2]) / stock1_data['Close'][-2]
        stock2_pct_change = (stock2_data['Close'][-1] - stock2_data['Close'][-2]) / stock2_data['Close'][-2]

        potential_return = stock1_pct_change - stock2_pct_change

        return potential_return
    except Exception as e:
        logger.error(f"Error calculating return for {stock1} and {stock2}: {e}\n{traceback.format_exc()}")
        return None

def predict_price(stock: str, data: pd.DataFrame) -> float:
    try:
        stock_data = data[stock]

        model = ARIMA(stock_data['Close'], order=(5,1,0))
        model_fit = model.fit(disp=0)

        prediction = model_fit.forecast()[0]

        return prediction
    except Exception as e:
        logger.error(f"Error predicting price for {stock}: {e}\n{traceback.format_exc()}")
        return 0

def worker_process(args: tuple):
    (pair, data, similarity_matrix) = args
    stock1, stock2 = pair
    potential_return = calculate_pairs_return(stock1, stock2, data)
    if potential_return is not None:
        stock1_prediction = predict_price(stock1, data)
        stock2_prediction = predict_price(stock2, data)
        if stock1_prediction is not None and stock2_prediction is not None:
            predicted_return = (stock1_prediction - data[stock1]['Close'][-1]) / data[stock1]['Close'][-1] - \
                               (stock2_prediction - data[stock2]['Close'][-1]) / data[stock2]['Close'][-1]
            weighted_return = predicted_return * similarity_matrix[pair[0]][pair[1]]
            return (weighted_return, (stock1, stock2))
    return None

def get_best_pair(file: str) -> tuple:
    df = pd.read_csv(file)

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

    best_pair = None, None
    highest_weighted_return = None

    grouped = df.groupby('Cluster')

    for name, group in grouped:
        if len(group) < 2: continue

        similarity_matrix = cosine_similarity(tfidf_matrix[group.index])
        stocks = group['stock_name'].values
        data = yf.download(stocks.tolist(), period="1d")
        pairs = [(i, j) for i in range(len(stocks)) for j in range(i+1, len(stocks))]

        with Pool() as pool:
            results = pool.map(worker_process, [(pair, data, similarity_matrix) for pair in pairs])

        for result in results:
            if result is not None and (highest_weighted_return is None or result[0] > highest_weighted_return):
                highest_weighted_return = result[0]
                best_pair = result[1]

    return best_pair

