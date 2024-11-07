# utils.py
import pandas as pd
import numpy as np
import pytz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb
from polygon import RESTClient
from datetime import datetime, timedelta, time, date
from pandas.tseries.offsets import BDay
import os
from dotenv import load_dotenv 
from ta import trend, momentum

#######################################################################################

def calculate_indicators(data):

    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    

    def macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
        ema_fast = ema(series, fastperiod)
        ema_slow = ema(series, slowperiod)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signalperiod)
        return macd_line, signal_line

    data['RSI'] = rsi(data['close'], period=14)
    data['EMA_Fast'] = ema(data['close'], period=3)
    data['EMA_Slow'] = ema(data['close'], period=9)
    data['MACD'], data['MACD_Signal'] = macd(data['close'])
    envelope_length = 21
    envelope_percent = 0.3 / 100
    data['Envelope_Upper'] = ema(data['close'], period=envelope_length) * (1 + envelope_percent)
    data['Envelope_Lower'] = ema(data['close'], period=envelope_length) * (1 - envelope_percent)
    return data

def calculate_scores_for_real_trade(data, max_buy_score,max_sell_score):
    data['score_ema'] = 0
    data['score_macd'] = 0
    data['score_rsi30'] = 0
    data['score_envelope'] = 0
    data['score_bullCond'] = 0
    data['sell_score_ema'] = 0
    data['sell_score_macd'] = 0
    data['sell_score_rsi70'] = 0
    data['sell_score_envelope'] = 0
    data['sell_score_bearCond'] = 0
    
    for i in range(1, len(data)):
        if (data['EMA_Fast'].iloc[i] > data['EMA_Slow'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] <= data['EMA_Slow'].iloc[i-1]):
            data.at[i, 'score_ema'] = 100
        else:
            data.at[i, 'score_ema'] = max(0, data.at[i-1, 'score_ema'] - 10)
        
        if (data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and 
            data['MACD'].iloc[i] < 0 and 
            data['MACD'].iloc[i-1] <= data['MACD_Signal'].iloc[i-1]):
            data.at[i, 'score_macd'] = 100
        else:
            data.at[i, 'score_macd'] = max(0, data.at[i-1, 'score_macd'] - 10)
        
        if (data['RSI'].iloc[i] > 30 and 
            data['RSI'].iloc[i-1] <= 30):
            data.at[i, 'score_rsi30'] = 100
        else:
            data.at[i, 'score_rsi30'] = max(0, data.at[i-1, 'score_rsi30'] - 10)
        
        if (data['EMA_Fast'].iloc[i] > data['Envelope_Lower'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] <= data['Envelope_Lower'].iloc[i-1]):
            data.at[i, 'score_envelope'] = 100
        else:
            data.at[i, 'score_envelope'] = max(0, data.at[i-1, 'score_envelope'] - 10)
        
        if (data['EMA_Fast'].iloc[i] < data['EMA_Slow'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] >= data['EMA_Slow'].iloc[i-1]):
            data.at[i, 'sell_score_ema'] = 100
        else:
            data.at[i, 'sell_score_ema'] = max(0, data.at[i-1, 'sell_score_ema'] - 10)
        
        if (data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and 
            data['MACD'].iloc[i] > 0 and 
            data['MACD'].iloc[i-1] >= data['MACD_Signal'].iloc[i-1]):
            data.at[i, 'sell_score_macd'] = 100
        else:
            data.at[i, 'sell_score_macd'] = max(0, data.at[i-1, 'sell_score_macd'] - 10)
        
        if (data['RSI'].iloc[i] < 70 and 
            data['RSI'].iloc[i-1] >= 70):
            data.at[i, 'sell_score_rsi70'] = 100
        else:
            data.at[i, 'sell_score_rsi70'] = max(0, data.at[i-1, 'sell_score_rsi70'] - 10)
        
        if (data['EMA_Fast'].iloc[i] < data['Envelope_Upper'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] >= data['Envelope_Upper'].iloc[i-1]):
            data.at[i, 'sell_score_envelope'] = 100
        else:
            data.at[i, 'sell_score_envelope'] = max(0, data.at[i-1, 'sell_score_envelope'] - 10)
    
    data['score'] = data['score_ema'] + data['score_macd'] + data['score_rsi30'] + data['score_envelope'] + data['score_bullCond']
    data['sell_score'] = data['sell_score_ema'] + data['sell_score_macd'] + data['sell_score_rsi70'] + data['sell_score_envelope'] + data['sell_score_bearCond']
    
    data['total_percent'] = (data['score'] / max_buy_score) * 100
    data['total_sell_percent'] = (data['sell_score'] / max_sell_score) * 100
    
    return data

def calculate_scores_for_train(data, max_buy_score,max_sell_score):
    data['score_ema'] = 0
    data['score_macd'] = 0
    data['score_rsi30'] = 0
    data['score_envelope'] = 0
    data['score_bullCond'] = 0
    data['sell_score_ema'] = 0
    data['sell_score_macd'] = 0
    data['sell_score_rsi70'] = 0
    data['sell_score_envelope'] = 0
    data['sell_score_bearCond'] = 0
    
    for i in range(1, len(data)):
        # เงื่อนไขการซื้อ
        if (data['EMA_Fast'].iloc[i] > data['EMA_Slow'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] <= data['EMA_Slow'].iloc[i-1]):
            data.at[data.index[i], 'score_ema'] = 100
        else:
            data.at[data.index[i], 'score_ema'] = max(0, data.at[data.index[i-1], 'score_ema'] - 10)
        
        if (data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and 
            data['MACD'].iloc[i] < 0 and 
            data['MACD'].iloc[i-1] <= data['MACD_Signal'].iloc[i-1]):
            data.at[data.index[i], 'score_macd'] = 100
        else:
            data.at[data.index[i], 'score_macd'] = max(0, data.at[data.index[i-1], 'score_macd'] - 10)
        
        if (data['RSI'].iloc[i] > 30 and 
            data['RSI'].iloc[i-1] <= 30):
            data.at[data.index[i], 'score_rsi30'] = 100
        else:
            data.at[data.index[i], 'score_rsi30'] = max(0, data.at[data.index[i-1], 'score_rsi30'] - 10)
        
        if (data['EMA_Fast'].iloc[i] > data['Envelope_Lower'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] <= data['Envelope_Lower'].iloc[i-1]):
            data.at[data.index[i], 'score_envelope'] = 100
        else:
            data.at[data.index[i], 'score_envelope'] = max(0, data.at[data.index[i-1], 'score_envelope'] - 10)
        
        # เงื่อนไขการขาย
        if (data['EMA_Fast'].iloc[i] < data['EMA_Slow'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] >= data['EMA_Slow'].iloc[i-1]):
            data.at[data.index[i], 'sell_score_ema'] = 100
        else:
            data.at[data.index[i], 'sell_score_ema'] = max(0, data.at[data.index[i-1], 'sell_score_ema'] - 10)
        
        if (data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and 
            data['MACD'].iloc[i] > 0 and 
            data['MACD'].iloc[i-1] >= data['MACD_Signal'].iloc[i-1]):
            data.at[data.index[i], 'sell_score_macd'] = 100
        else:
            data.at[data.index[i], 'sell_score_macd'] = max(0, data.at[data.index[i-1], 'sell_score_macd'] - 10)
        
        if (data['RSI'].iloc[i] < 70 and 
            data['RSI'].iloc[i-1] >= 70):
            data.at[data.index[i], 'sell_score_rsi70'] = 100
        else:
            data.at[data.index[i], 'sell_score_rsi70'] = max(0, data.at[data.index[i-1], 'sell_score_rsi70'] - 10)
        
        if (data['EMA_Fast'].iloc[i] < data['Envelope_Upper'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] >= data['Envelope_Upper'].iloc[i-1]):
            data.at[data.index[i], 'sell_score_envelope'] = 100
        else:
            data.at[data.index[i], 'sell_score_envelope'] = max(0, data.at[data.index[i-1], 'sell_score_envelope'] - 10)
    
    data['score'] = data['score_ema'] + data['score_macd'] + data['score_rsi30'] + data['score_envelope'] + data['score_bullCond']
    data['sell_score'] = data['sell_score_ema'] + data['sell_score_macd'] + data['sell_score_rsi70'] + data['sell_score_envelope'] + data['sell_score_bearCond']
    
    data['total_percent'] = (data['score'] / max_buy_score) * 100
    data['total_sell_percent'] = (data['sell_score'] / max_sell_score) * 100
    
    return data

def calculate_scores_for_test(data, max_buy_score,max_sell_score):
    data['score_ema'] = 0
    data['score_macd'] = 0
    data['score_rsi30'] = 0
    data['score_envelope'] = 0
    data['score_bullCond'] = 0
    data['sell_score_ema'] = 0
    data['sell_score_macd'] = 0
    data['sell_score_rsi70'] = 0
    data['sell_score_envelope'] = 0
    data['sell_score_bearCond'] = 0
    
    for i in range(1, len(data)):
        # เงื่อนไขการซื้อ
        if (data['EMA_Fast'].iloc[i] > data['EMA_Slow'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] <= data['EMA_Slow'].iloc[i-1]):
            data.at[data.index[i], 'score_ema'] = 100
        else:
            data.at[data.index[i], 'score_ema'] = max(0, data.at[data.index[i-1], 'score_ema'] - 10)
        
        if (data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and 
            data['MACD'].iloc[i] < 0 and 
            data['MACD'].iloc[i-1] <= data['MACD_Signal'].iloc[i-1]):
            data.at[data.index[i], 'score_macd'] = 100
        else:
            data.at[data.index[i], 'score_macd'] = max(0, data.at[data.index[i-1], 'score_macd'] - 10)
        
        if (data['RSI'].iloc[i] > 30 and 
            data['RSI'].iloc[i-1] <= 30):
            data.at[data.index[i], 'score_rsi30'] = 100
        else:
            data.at[data.index[i], 'score_rsi30'] = max(0, data.at[data.index[i-1], 'score_rsi30'] - 10)
        
        if (data['EMA_Fast'].iloc[i] > data['Envelope_Lower'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] <= data['Envelope_Lower'].iloc[i-1]):
            data.at[data.index[i], 'score_envelope'] = 100
        else:
            data.at[data.index[i], 'score_envelope'] = max(0, data.at[data.index[i-1], 'score_envelope'] - 10)
        
        # เงื่อนไขการขาย
        if (data['EMA_Fast'].iloc[i] < data['EMA_Slow'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] >= data['EMA_Slow'].iloc[i-1]):
            data.at[data.index[i], 'sell_score_ema'] = 100
        else:
            data.at[data.index[i], 'sell_score_ema'] = max(0, data.at[data.index[i-1], 'sell_score_ema'] - 10)
        
        if (data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and 
            data['MACD'].iloc[i] > 0 and 
            data['MACD'].iloc[i-1] >= data['MACD_Signal'].iloc[i-1]):
            data.at[data.index[i], 'sell_score_macd'] = 100
        else:
            data.at[data.index[i], 'sell_score_macd'] = max(0, data.at[data.index[i-1], 'sell_score_macd'] - 10)
        
        if (data['RSI'].iloc[i] < 70 and 
            data['RSI'].iloc[i-1] >= 70):
            data.at[data.index[i], 'sell_score_rsi70'] = 100
        else:
            data.at[data.index[i], 'sell_score_rsi70'] = max(0, data.at[data.index[i-1], 'sell_score_rsi70'] - 10)
        
        if (data['EMA_Fast'].iloc[i] < data['Envelope_Upper'].iloc[i] and 
            data['EMA_Fast'].iloc[i-1] >= data['Envelope_Upper'].iloc[i-1]):
            data.at[data.index[i], 'sell_score_envelope'] = 100
        else:
            data.at[data.index[i], 'sell_score_envelope'] = max(0, data.at[data.index[i-1], 'sell_score_envelope'] - 10)
    
    data['score'] = data['score_ema'] + data['score_macd'] + data['score_rsi30'] + data['score_envelope'] + data['score_bullCond']
    data['sell_score'] = data['sell_score_ema'] + data['sell_score_macd'] + data['sell_score_rsi70'] + data['sell_score_envelope'] + data['sell_score_bearCond']
    
    data['total_percent'] = (data['score'] / max_buy_score) * 100
    data['total_sell_percent'] = (data['sell_score'] / max_sell_score) * 100
    
    return data

def generate_signals_for_real_trade(data, buy_score, sell_score, capital):
    data['Buy_Signal'] = False
    data['Sell_Signal'] = False
    position = False
    shares = 0
    entry_price = 0.0
    trailing_stop = False 
    
    initial_capital = capital
    data['capital'] = capital
    data['equity'] = capital
    
    market_open_time = pd.to_datetime('09:30:00').time()
    market_close_time = pd.to_datetime('15:45:00').time()
    
    for i in range(len(data)):
        current_time = data['timestamp'].iloc[i].time()
        current_date = data['timestamp'].iloc[i].date()
        
        in_market_hours = market_open_time <= current_time <= market_close_time
        
        current_close = data['close'].iloc[i]
        
        if position:
            if current_close <= entry_price * 0.99:
                data.at[i, 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
            else:
                if not trailing_stop and current_close >= entry_price * 1.005:
                    trailing_stop = True
                    stop_loss = entry_price  
                    
                if trailing_stop:
                    if current_close <= stop_loss:
                        data.at[i, 'Sell_Signal'] = True
                        capital += shares * current_close
                        shares = 0
                        position = False
                        entry_price = 0.0
                        trailing_stop = False  
                

                if current_close >= entry_price * 1.02:
                    data.at[i, 'Sell_Signal'] = True
                    capital += shares * current_close
                    shares = 0
                    position = False
                    entry_price = 0.0
                    trailing_stop = False  
        
        # เงื่อนไขการซื้อ
        buy_condition = (
            data['total_percent'].iloc[i] > buy_score and  # ลดจาก 50 เป็น 40
            not position and
            in_market_hours
        )
        
        # เงื่อนไขการขาย
        sell_condition = (
            data['total_sell_percent'].iloc[i] > sell_score and
            position and
            in_market_hours
        )
        
        if buy_condition:
            # คำนวณจำนวนหุ้นที่สามารถซื้อได้
            shares = capital // current_close  
            if shares > 0:
                data.at[i, 'Buy_Signal'] = True
                capital -= shares * current_close 
                position = True
                entry_price = current_close  
                trailing_stop = False  
        elif sell_condition:
            if shares > 0:
                # ทำการขายทั้งหมด
                data.at[i, 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
        
        if current_time == market_close_time:
            if position and shares > 0:
                data.at[i, 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False 
        
        equity = capital + shares * current_close
        data.at[i, 'capital'] = capital
        data.at[i, 'equity'] = equity
    
    return data

def generate_signals_for_train(data, buy_score, sell_score, capital):

    data['Buy_Signal'] = False
    data['Sell_Signal'] = False 
    position = False
    shares = 0
    entry_price = 0.0
    trailing_stop = False  
    initial_capital = capital
    data['capital'] = capital
    data['equity'] = capital
    
    market_open_time = pd.to_datetime('09:30:00').time()
    market_close_time = pd.to_datetime('15:45:00').time()
    
    for i in range(len(data)):
        current_time = data['timestamp'].iloc[i].time()
        current_date = data['timestamp'].iloc[i].date()
        
        in_market_hours = market_open_time <= current_time <= market_close_time
        
        current_close = data['close'].iloc[i]
        
        if position:
            if current_close <= entry_price * 0.99:
                data.at[data.index[i], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
            else:
                if not trailing_stop and current_close >= entry_price * 1.005:
                    trailing_stop = True
                    stop_loss = entry_price  
                
                if trailing_stop:
                    if current_close <= stop_loss:
                        data.at[data.index[i], 'Sell_Signal'] = True
                        capital += shares * current_close
                        shares = 0
                        position = False
                        entry_price = 0.0
                        trailing_stop = False  
                

                if current_close >= entry_price * 1.01:
                    data.at[data.index[i], 'Sell_Signal'] = True
                    capital += shares * current_close
                    shares = 0
                    position = False
                    entry_price = 0.0
                    trailing_stop = False  
        
        # เงื่อนไขการซื้อ
        buy_condition = (
            data['total_percent'].iloc[i] > buy_score and  
            not position and
            in_market_hours
        )
        
        # เงื่อนไขการขาย
        sell_condition = (
            data['total_sell_percent'].iloc[i] > sell_score and
            position and
            in_market_hours
        )
        
        if buy_condition:
            # คำนวณจำนวนหุ้นที่สามารถซื้อได้
            shares = capital // current_close  
            if shares > 0:
                data.at[data.index[i], 'Buy_Signal'] = True
                capital -= shares * current_close 
                position = True
                entry_price = current_close  
                trailing_stop = False  
        elif sell_condition:
            if shares > 0:
                # ทำการขายทั้งหมด
                data.at[data.index[i], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
        
        if current_time == market_close_time:
            if position and shares > 0:
                data.at[data.index[i], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
        
        equity = capital + shares * current_close
        data.at[data.index[i], 'capital'] = capital
        data.at[data.index[i], 'equity'] = equity
    
    return data

def generate_signals_for_test(data):

    data['Buy_Signal'] = False
    data['Sell_Signal'] = False
    buy_score = 25
    sell_score = 50
    
    position = False
    shares = 0
    entry_price = 0.0
    trailing_stop = False  
    
    # เงินทุนเริ่มต้น
    capital = 80000.0  
    initial_capital = capital
    data['capital'] = capital
    data['equity'] = capital
    
    market_open_time = time(9, 30)
    market_close_time = time(15, 45)
    
    for i in range(len(data)):
        current_time = data.index[i].time()
        current_date = data.index[i].date()
        
        in_market_hours = market_open_time <= current_time <= market_close_time
        
        current_close = data['close'].iloc[i]
        
        if position:
            if current_close <= entry_price * 0.99:
                data.at[data.index[i], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
            else:
                if not trailing_stop and current_close >= entry_price * 1.005:
                    trailing_stop = True
                    stop_loss = entry_price  
                
                if trailing_stop:
                    if current_close <= stop_loss:
                        data.at[data.index[i], 'Sell_Signal'] = True
                        capital += shares * current_close
                        shares = 0
                        position = False
                        entry_price = 0.0
                        trailing_stop = False  
                

                if current_close >= entry_price * 1.01:
                    data.at[data.index[i], 'Sell_Signal'] = True
                    capital += shares * current_close
                    shares = 0
                    position = False
                    entry_price = 0.0
                    trailing_stop = False  
        
        # เงื่อนไขการซื้อ
        buy_condition = (
            data['total_percent'].iloc[i] > buy_score and  
            not position and
            in_market_hours
        )
        
        # เงื่อนไขการขาย
        sell_condition = (
            data['total_sell_percent'].iloc[i] > sell_score and
            position and
            in_market_hours
        )
        
        if buy_condition:
            # คำนวณจำนวนหุ้นที่สามารถซื้อได้
            shares = capital // current_close  
            if shares > 0:
                data.at[data.index[i], 'Buy_Signal'] = True
                capital -= shares * current_close 
                position = True
                entry_price = current_close  
                trailing_stop = False  
        elif sell_condition:
            if shares > 0:
                
                data.at[data.index[i], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
        
        if current_time == market_close_time:
            if position and shares > 0:
                data.at[data.index[i], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  
        
        equity = capital + shares * current_close
        data.at[data.index[i], 'capital'] = capital
        data.at[data.index[i], 'equity'] = equity
    
    return data

def create_labels(data, profit_p, loss_p):
    data['Label'] = pd.NA
    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i]:
            entry_price = data['close'].iloc[i]
            sell_signals_after_buy = data.iloc[i+1:][data['Sell_Signal'] == True]
            
            if not sell_signals_after_buy.empty:
                sell_index = sell_signals_after_buy.index[0]
                exit_price = data.loc[sell_index, 'close']
                profit_loss_percent = (exit_price - entry_price) / entry_price * 100
                
                if profit_loss_percent >= profit_p:
                    data.at[i, 'Label'] = 1
                elif profit_loss_percent <= loss_p:
                    data.at[i, 'Label'] = 0
            else:
                data.at[i, 'Label'] = pd.NA  
    labeled_data = data.dropna(subset=['Label'])
    labeled_data['Label'] = labeled_data['Label'].astype(int)   
    return labeled_data

def polygon_request(ticker, multiplier, timespan, start, end):
    API_KEY = os.getenv('API_KEY')  
    client = RESTClient(API_KEY) 
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=start, to=end, limit=50000):
        aggs.append(a)
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
    df.set_index('timestamp', inplace=True)
    df.drop(['transactions', 'otc'], axis=1, inplace=True)
    return df

def get_stock_data(stockName, timeFrame, unit, todate=None, BD=150):
    if todate is None:
        todate = datetime.now().strftime('%Y-%m-%d')
    else:
        try:
            print(f'todate is {todate}')
            datetime.strptime(todate, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect date format for todate, should be YYYY-MM-DD")

    todate_dt = pd.to_datetime(todate)
    fromDate = todate_dt - BDay(BD)
    
    print(f'get data from polygon from {fromDate.strftime("%Y-%m-%d")} to {todate_dt} ..........')
    
    if unit == 'second':
        timespan = 'second'
    elif unit == 'minute':
        timespan = 'minute'
    else:
        raise ValueError("unit ต้องเป็น 'second' หรือ 'minute' เท่านั้น")
    
    data = polygon_request(stockName, timeFrame, timespan, fromDate.strftime('%Y-%m-%d'), todate)
    
    data = data.reset_index()
    
    if not pd.api.types.is_datetime64tz_dtype(data['timestamp']):
        data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        data['timestamp'] = data['timestamp'].dt.tz_convert('US/Eastern')
    
    data = data.sort_values(by='timestamp').reset_index(drop=True)
    
    return data

def filter_market_hours(data):
    market_open_time = time(9, 30)
    market_close_time = time(16, 0)
    return data.between_time(market_open_time, market_close_time)

def predict_new_data(file_path, model, scaler):
    new_data = file_path
    
    if 'timestamp' not in new_data.columns:   
        new_data = new_data.reset_index()
        new_data = new_data.rename(columns={'index': 'timestamp'})
    
    
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])     
    if not pd.api.types.is_datetime64tz_dtype(new_data['timestamp']):       
        new_data['timestamp'] = new_data['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:       
        new_data['timestamp'] = new_data['timestamp'].dt.tz_convert('US/Eastern')
    
    new_data = new_data.sort_values(by='timestamp').reset_index(drop=True)   
    new_data.set_index('timestamp', inplace=True)   
    new_data = calculate_indicators(new_data) 
    new_data = calculate_scores_for_test(new_data)
    new_data = generate_signals_for_test(new_data)
    
    features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal', 
                'Envelope_Upper', 'Envelope_Lower', 'total_percent','total_sell_percent']
    
    buy_signals_new = new_data[new_data['Buy_Signal']]
    
    if buy_signals_new.empty:
        print("ไม่มี Buy_Signal ในข้อมูลใหม่")
        predicted_data = pd.DataFrame()  
    else:
        X_new = buy_signals_new[features]
        X_new_scaled = scaler.transform(X_new)
        predicted_labels = model.predict(X_new_scaled)
        predicted_data = buy_signals_new.copy()
        predicted_data['Predicted_Label'] = predicted_labels
        print(predicted_data[['close', 'Predicted_Label']])

    return new_data, predicted_data

#######################################################################################