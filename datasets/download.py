import yfinance as yf
import os
proxy = 'http://' 
os.environ['HTTP_PROXY'] = proxy 
os.environ['HTTPS_PROXY'] = proxy
start_date = '2015-01-01'
end_date = '2024-01-01'
FTSE_data = yf.download("^FVX", start=start_date, end=end_date)
print(FTSE_data.head())

FTSE_data.to_csv('FVX.csv')
