import requests
import numpy as np
import matplotlib.pyplot as plt

class MarketAnalysis:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'http://api.marketstack.com/v1/'

    def get_eod_closing_prices(self, ticker, start_date, end_date):
        endpoint = 'eod'
        params = {
            'access_key': self.api_key,
            'symbols': ticker,
            'date_from': start_date,
            'date_to': end_date
        }
        url = self.base_url + endpoint
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
        
        data = response.json()
        
        if 'data' not in data:
            raise Exception(f"Unexpected response format: {data}")
        
        records = data['data']
        
        # Sort records by date
        sorted_records = sorted(records, key=lambda x: x['date'])
        
        # Extract the closing prices into a one-dimensional list
        closing_prices = [record['close'] for record in sorted_records]
        return closing_prices

    def calculate_daily_percentage_growth(self, closing_prices):
        # Calculate daily percentage growth
        percentage_growth = []
        for i in range(1, len(closing_prices)):
            growth = ((closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]) * 100
            percentage_growth.append(growth)
        return percentage_growth
    
    def create_11_dimensional_vectors(self, percentage_growth):
        input_vector = []
        output_vector = []
        for i in range(len(percentage_growth) - 12):
            # Create 11-dimensional vector
            input_vector.append(percentage_growth[i:i+10])
            output_vector.append(sum(percentage_growth[i+10:i+13]))
        return np.array(input_vector), np.array(output_vector)


# Market data retrieval and analysis
api_key = 'e8b2a8a00868eda4e48239f1bbb43b91'
ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2023-06-01'

market_analysis = MarketAnalysis(api_key)
closing_prices = market_analysis.get_eod_closing_prices(ticker, start_date, end_date)
percentage_growth = market_analysis.calculate_daily_percentage_growth(closing_prices)
input_vector, output_vector = market_analysis.create_11_dimensional_vectors(percentage_growth)

