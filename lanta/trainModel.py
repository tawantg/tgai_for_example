import pandas as pd
import numpy as np
import mplfinance as mpf
import pytz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb

from datetime import datetime, timedelta
from datetime import date

import os
from dotenv import load_dotenv 



# ฟังก์ชั่นคำนวณ EMA
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ฟังก์ชั่นคำนวณ RSI
def rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ฟังก์ชั่นคำนวณ MACD
def macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = ema(series, fastperiod)
    ema_slow = ema(series, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signalperiod)
    return macd_line, signal_line

# ฟังก์ชั่นคำนวณตัวชี้วัดต่างๆ
def calculate_indicators(data):
    data['RSI'] = rsi(data['close'], period=14)
    data['EMA_Fast'] = ema(data['close'], period=3)
    data['EMA_Slow'] = ema(data['close'], period=9)
    data['MACD'], data['MACD_Signal'] = macd(data['close'])
    
    # คำนวณ Envelope
    envelope_length = 21
    envelope_percent = 0.3 / 100
    data['Envelope_Upper'] = ema(data['close'], period=envelope_length) * (1 + envelope_percent)
    data['Envelope_Lower'] = ema(data['close'], period=envelope_length) * (1 - envelope_percent)
    
    return data

# ฟังก์ชั่นคำนวณคะแนนซื้อและขาย
def calculate_scores(data):
    # เริ่มต้นคอลัมน์สำหรับคะแนน
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
    
    # กำหนดค่ามากสุดของคะแนน
    max_buy_score = 600
    max_sell_score = 600
    
    # วนลูปผ่านแต่ละแถวเพื่อกำหนดคะแนน
    for i in range(1, len(data)):
        # เงื่อนไขการซื้อ
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
        
        # เงื่อนไขการขาย
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
    
    # คำนวณคะแนนรวมและเปอร์เซ็นต์
    data['score'] = data['score_ema'] + data['score_macd'] + data['score_rsi30'] + data['score_envelope'] + data['score_bullCond']
    data['sell_score'] = data['sell_score_ema'] + data['sell_score_macd'] + data['sell_score_rsi70'] + data['sell_score_envelope'] + data['sell_score_bearCond']
    
    data['total_percent'] = (data['score'] / max_buy_score) * 100
    data['total_sell_percent'] = (data['sell_score'] / max_sell_score) * 100
    
    return data

def generate_signals(data):

    data['Buy_Signal'] = False
    data['Sell_Signal'] = False
    buy_score = 25
    sell_score =25
    

    position = False
    shares = 0
    entry_price = 0.0
    trailing_stop = False 
    
    # เงินทุนเริ่มต้น
    capital = 80000.0  
    initial_capital = capital
    data['capital'] = capital
    data['equity'] = capital
    
    # กำหนดช่วงเวลาทำการตลาด US/Eastern Time
    market_open_time = pd.to_datetime('09:30:00').time()
    market_close_time = pd.to_datetime('15:45:00').time()
    
    # วนลูปผ่านแต่ละแถวเพื่อกำหนดสัญญาณซื้อขายและบริหารเงินทุน
    for i in range(len(data)):
        current_time = data['timestamp'].iloc[i].time()
        current_date = data['timestamp'].iloc[i].date()
        
        # เช็คว่าอยู่ในช่วงเวลาทำการตลาดหรือไม่
        in_market_hours = market_open_time <= current_time <= market_close_time
        
        current_close = data['close'].iloc[i]
        
        # ถ้ามีตำแหน่งเปิดอยู่ ต้องตรวจสอบการตัดขาดทุน
        if position:
            # ตรวจสอบว่าราคาลดลง 1% จากราคาที่ซื้อหรือไม่
            if current_close <= entry_price * 0.99:
                # ทำการขายทั้งหมดเพื่อตัดขาดทุน
                data.at[i, 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  # รีเซ็ต trailing stop
            else:
                # ตรวจสอบว่ากำไรได้มากกว่า 0.5% จากราคาที่ซื้อหรือไม่
                if not trailing_stop and current_close >= entry_price * 1.005:
                    trailing_stop = True
                    stop_loss = entry_price  # เลื่อนจุดตัดขาดทุนไปที่ราคาซื้อ
                    
                # ถ้าได้เลื่อนจุดตัดขาดทุนแล้ว ให้ตรวจสอบราคาปัจจุบันว่าลดลงถึง stop_loss หรือไม่
                if trailing_stop:
                    if current_close <= stop_loss:
                        # ทำการขายทั้งหมดเพื่อตัดขาดทุน
                        data.at[i, 'Sell_Signal'] = True
                        capital += shares * current_close
                        shares = 0
                        position = False
                        entry_price = 0.0
                        trailing_stop = False  # รีเซ็ต trailing stop
                
                # ตรวจสอบว่ากำไรได้ 2% หรือไม่
                if current_close >= entry_price * 1.01:
                    # ทำการขายเพื่อรับกำไร
                    data.at[i, 'Sell_Signal'] = True
                    capital += shares * current_close
                    shares = 0
                    position = False
                    entry_price = 0.0
                    trailing_stop = False  # รีเซ็ต trailing stop
        
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
            shares = capital // current_close  # ใช้ // เพื่อให้ได้จำนวนเต็ม
            if shares > 0:
                data.at[i, 'Buy_Signal'] = True
                capital -= shares * current_close  # หักเงินทุน
                position = True
                entry_price = current_close  # บันทึกราคาที่ซื้อ
                trailing_stop = False  # รีเซ็ต trailing stop
        elif sell_condition:
            if shares > 0:
                # ทำการขายทั้งหมด
                data.at[i, 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  # รีเซ็ต trailing stop
        
        # เช็คว่าถึงเวลาปิดตลาดหรือไม่ เพื่อขายทั้งหมด
        if current_time == market_close_time:
            if position and shares > 0:
                data.at[i, 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False  # รีเซ็ต trailing stop
        
        # คำนวณ equity (เงินทุนรวม)
        equity = capital + shares * current_close
        data.at[i, 'capital'] = capital
        data.at[i, 'equity'] = equity
    
    return data

# ฟังก์ชั่นสร้าง Label สำหรับ Machine Learning
def create_labels(data):
    # เพิ่มคอลัมน์ Label เริ่มต้นเป็น NaN
    data['Label'] = pd.NA
    
    # วนลูปผ่านแต่ละแถวเพื่อกำหนด Label
    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i]:
            entry_price = data['close'].iloc[i]
            # ค้นหาจุด Sell_Signal หลังจาก Buy_Signal
            sell_signals_after_buy = data.iloc[i+1:][data['Sell_Signal'] == True]
            
            if not sell_signals_after_buy.empty:
                # หาจุดขายแรกหลังจากการซื้อ
                sell_index = sell_signals_after_buy.index[0]
                exit_price = data.loc[sell_index, 'close']
                profit_loss_percent = (exit_price - entry_price) / entry_price * 100
                
                if profit_loss_percent >= 0.1:
                    data.at[i, 'Label'] = 1
                elif profit_loss_percent <= -0.1:
                    data.at[i, 'Label'] = 0
                # ถ้าไม่ตรงเงื่อนไขให้เป็น NaN
            else:
                # ไม่มี Sell Signal หลังจากนี้
                data.at[i, 'Label'] = pd.NA  # หรืออาจกำหนดเป็น 0 หรือ 1 ตามความเหมาะสม
    
    # ลบแถวที่ Label เป็น NaN
    labeled_data = data.dropna(subset=['Label'])
    
    # แปลงประเภทข้อมูลของ Label ให้เป็น Integer
    labeled_data['Label'] = labeled_data['Label'].astype(int)
    
    return labeled_data

# ฟังก์ชั่นพล็อตกราฟแท่งเทียนพร้อมสัญญาณซื้อขายและ Label
def plot_signals(data, labeled_data, predicted_data=None):
    # เตรียมข้อมูลสำหรับการ plot หลังจากสร้าง Label แล้ว
    data['Date'] = data['timestamp'].dt.date  # สร้างคอลัมน์วันที่
    data.set_index('timestamp', inplace=True)
    plot_data = data[['open', 'high', 'low', 'close', 'volume']]
    
    # กำหนดจุดที่เป็นสัญญาณซื้อและขาย
    buy_signals = data[data['Buy_Signal']]
    sell_signals = data[data['Sell_Signal']]
    
    # เพิ่มการสร้าง Marker สำหรับ Label จากข้อมูลฝึกโมเดล
    # สร้าง DataFrame สำหรับ Label = 1 และ Label = 0 จาก labeled_data
    label_up = labeled_data[labeled_data['Label'] == 1]
    label_down = labeled_data[labeled_data['Label'] == 0]
    
    # สร้าง DataFrame สำหรับการ plot ลูกศรจากข้อมูลฝึกโมเดล
    label_marker_data_up = pd.DataFrame(index=plot_data.index)
    label_marker_data_down = pd.DataFrame(index=plot_data.index)
    
    label_marker_data_up['Label_Up'] = label_up['low'] - 0.2
    label_marker_data_down['Label_Down'] = label_down['high'] + 0.2
    
    # สร้าง DataFrame สำหรับการ plot ลูกศรจากข้อมูลทำนายใหม่
    if predicted_data is not None:
        predicted_marker_up = pd.DataFrame(index=predicted_data.index)
        predicted_marker_down = pd.DataFrame(index=predicted_data.index)
        
        predicted_up = predicted_data[predicted_data['Predicted_Label'] == 1]
        predicted_down = predicted_data[predicted_data['Predicted_Label'] == 0]
        
        predicted_marker_up['Predicted_Label_Up'] = predicted_up['low'] - 0.3  # ปรับตำแหน่งลูกศรตามต้องการ
        predicted_marker_down['Predicted_Label_Down'] = predicted_down['high'] + 0.3
    
    # วนลูปผ่านแต่ละกลุ่มวันและพล็อตกราฟ
    grouped = data.groupby('Date')
    
    for date, group in grouped:
        if group.empty:
            continue
        
        # เตรียมข้อมูลสำหรับการ plot
        group_plot = group[['open', 'high', 'low', 'close', 'volume']].copy()
        group_plot.index = group.index  # ใช้ timestamp เป็น index
        
        # กำหนดสัญญาณซื้อและขายสำหรับวันนั้น
        group_buy_signals = group[group['Buy_Signal']]
        group_sell_signals = group[group['Sell_Signal']]
        group_label_up = group[group['Label'] == 1]
        group_label_down = group[group['Label'] == 0]
        
        # สร้าง DataFrame สำหรับสัญญาณซื้อและขาย
        group_buy_marker = pd.DataFrame(index=group_plot.index)
        group_sell_marker = pd.DataFrame(index=group_plot.index)
        group_label_marker_up = pd.DataFrame(index=group_plot.index)
        group_label_marker_down = pd.DataFrame(index=group_plot.index)
        
        if not group_buy_signals.empty:
            group_buy_marker['Buy'] = group_buy_signals['low'] - 0.1
        else:
            group_buy_marker['Buy'] = np.nan  # กำหนดเป็น NaN เมื่อไม่มี Buy_Signal
        
        if not group_sell_signals.empty:
            group_sell_marker['Sell'] = group_sell_signals['high'] + 0.1
        else:
            group_sell_marker['Sell'] = np.nan  # กำหนดเป็น NaN เมื่อไม่มี Sell_Signal
        
        if not group_label_up.empty:
            group_label_marker_up['Label_Up'] = group_label_up['low'] - 0.2
        else:
            group_label_marker_up['Label_Up'] = np.nan
        
        if not group_label_down.empty:
            group_label_marker_down['Label_Down'] = group_label_down['high'] + 0.2
        else:
            group_label_marker_down['Label_Down'] = np.nan
        
        # เตรียม addplots ถ้ามีสัญญาณซื้อหรือขายหรือ Label จากข้อมูลฝึกโมเดล
        add_plots = []
        if not group_buy_marker['Buy'].isnull().all():
            buy_markers = mpf.make_addplot(group_buy_marker['Buy'], type='scatter', markersize=100, marker='^', color='green')
            add_plots.append(buy_markers)
        
        if not group_sell_marker['Sell'].isnull().all():
            sell_markers = mpf.make_addplot(group_sell_marker['Sell'], type='scatter', markersize=100, marker='v', color='red')
            add_plots.append(sell_markers)
        
        if not group_label_marker_up['Label_Up'].isnull().all():
            label_up_markers_day = mpf.make_addplot(group_label_marker_up['Label_Up'], type='scatter', markersize=100, marker='^', color='blue')
            add_plots.append(label_up_markers_day)
        
        if not group_label_marker_down['Label_Down'].isnull().all():
            label_down_markers_day = mpf.make_addplot(group_label_marker_down['Label_Down'], type='scatter', markersize=100, marker='v', color='blue')
            add_plots.append(label_down_markers_day)
        
        # เพิ่มการสร้าง Marker สำหรับ Label จากข้อมูลทำนายใหม่
        if predicted_data is not None:
            # เลือกข้อมูลที่ตรงกับวันนั้น
            predicted_group = predicted_data[predicted_data['timestamp'].dt.date == date]
            if not predicted_group.empty:
                predicted_marker_up_day = pd.DataFrame(index=predicted_group.index)
                predicted_marker_down_day = pd.DataFrame(index=predicted_group.index)
                
                predicted_up_day = predicted_group[predicted_group['Predicted_Label'] == 1]
                predicted_down_day = predicted_group[predicted_group['Predicted_Label'] == 0]
                
                if not predicted_up_day.empty:
                    predicted_marker_up_day['Predicted_Label_Up'] = predicted_up_day['low'] - 0.3
                else:
                    predicted_marker_up_day['Predicted_Label_Up'] = np.nan
                
                if not predicted_down_day.empty:
                    predicted_marker_down_day['Predicted_Label_Down'] = predicted_down_day['high'] + 0.3
                else:
                    predicted_marker_down_day['Predicted_Label_Down'] = np.nan
                
                # เพิ่มสัญญาณทำนายลงใน add_plots
                if not predicted_marker_up_day['Predicted_Label_Up'].isnull().all():
                    pred_up_markers = mpf.make_addplot(predicted_marker_up_day['Predicted_Label_Up'], type='scatter', markersize=100, marker='^', color='purple')
                    add_plots.append(pred_up_markers)
                
                if not predicted_marker_down_day['Predicted_Label_Down'].isnull().all():
                    pred_down_markers = mpf.make_addplot(predicted_marker_down_day['Predicted_Label_Down'], type='scatter', markersize=100, marker='v', color='orange')
                    add_plots.append(pred_down_markers)
        
        # Plot กราฟแท่งเทียนพร้อมสัญญาณถ้ามี
        if add_plots:  # ตรวจสอบว่ามี add_plots หรือไม่
            mpf.plot(
                group_plot, 
                type='candle', 
                style='charles',
                addplot=add_plots,
                volume=True,
                title=f'Buy/Sell Signals and Labels on Candlestick Chart for {date}',
                ylabel='Price',
                ylabel_lower='Volume',
                tight_layout=True
            )
            plt.show()

def plot_signals_daily(data, labeled_data, predicted_data=None, date=None):
    # เตรียมข้อมูลสำหรับการ plot
    plot_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # กำหนดจุดที่เป็นสัญญาณซื้อและขาย
    buy_signals = data[data['Buy_Signal']]
    sell_signals = data[data['Sell_Signal']]
    
    # สร้าง DataFrame สำหรับสัญญาณซื้อและขาย
    buy_marker = pd.DataFrame(index=plot_data.index)
    sell_marker = pd.DataFrame(index=plot_data.index)
    label_marker_up = pd.DataFrame(index=plot_data.index)
    label_marker_down = pd.DataFrame(index=plot_data.index)
    
    buy_marker['Buy'] = np.where(data['Buy_Signal'], data['low'] - 0.1, np.nan)
    sell_marker['Sell'] = np.where(data['Sell_Signal'], data['high'] + 0.1, np.nan)
    
    if 'Label' in data.columns:
        label_marker_up['Label_Up'] = np.where(data['Label'] == 1, data['low'] - 0.2, np.nan)
        label_marker_down['Label_Down'] = np.where(data['Label'] == 0, data['high'] + 0.2, np.nan)
    
    # เตรียม addplots
    add_plots = []
    if not buy_marker['Buy'].isnull().all():
        add_plots.append(mpf.make_addplot(buy_marker['Buy'], type='scatter', markersize=100, marker='^', color='green'))
    
    if not sell_marker['Sell'].isnull().all():
        add_plots.append(mpf.make_addplot(sell_marker['Sell'], type='scatter', markersize=100, marker='v', color='red'))
    
    if not label_marker_up['Label_Up'].isnull().all():
        add_plots.append(mpf.make_addplot(label_marker_up['Label_Up'], type='scatter', markersize=100, marker='^', color='blue'))
    
    if not label_marker_down['Label_Down'].isnull().all():
        add_plots.append(mpf.make_addplot(label_marker_down['Label_Down'], type='scatter', markersize=100, marker='v', color='blue'))
    
    # เพิ่มการสร้าง Marker สำหรับ Label จากข้อมูลทำนายใหม่
    if predicted_data is not None:
        predicted_marker_up = pd.DataFrame(index=plot_data.index)
        predicted_marker_down = pd.DataFrame(index=plot_data.index)
        
        predicted_up = predicted_data[predicted_data['Predicted_Label'] == 1]
        predicted_down = predicted_data[predicted_data['Predicted_Label'] == 0]
        
        predicted_marker_up['Predicted_Label_Up'] = np.where(predicted_data['Predicted_Label'] == 1, data['low'] - 0.3, np.nan)
        predicted_marker_down['Predicted_Label_Down'] = np.where(predicted_data['Predicted_Label'] == 0, data['high'] + 0.3, np.nan)
        
        if not predicted_marker_up['Predicted_Label_Up'].isnull().all():
            add_plots.append(mpf.make_addplot(predicted_marker_up['Predicted_Label_Up'], type='scatter', markersize=100, marker='^', color='purple'))
        
        if not predicted_marker_down['Predicted_Label_Down'].isnull().all():
            add_plots.append(mpf.make_addplot(predicted_marker_down['Predicted_Label_Down'], type='scatter', markersize=100, marker='v', color='orange'))
    
    # Plot กราฟแท่งเทียนพร้อมสัญญาณ
    mpf.plot(
        plot_data, 
        type='candle', 
        style='charles',
        addplot=add_plots,
        volume=True,
        title=f'Buy/Sell Signals and Labels on Candlestick Chart for {date}',
        ylabel='Price',
        ylabel_lower='Volume',
        figsize=(20, 10),
        tight_layout=True
    )
    plt.show()

def plot_all_daily_charts(data, labeled_data, predicted_data=None):
    # เพิ่มคอลัมน์วันที่
    data['Date'] = data.index.date
    
    
    # วนลูปผ่านแต่ละวัน
    for date, group in data.groupby('Date'):
        print(f"Plotting chart for {date}")
        plot_signals_daily(group, labeled_data, predicted_data, date)


# data = get_stock_data(stockName, timeFrame, unit, todate)
data = pd.read_csv('dataForTest.csv')


stockName = 'COIN'
timeFrame = 15
unit = 'second'
todate = '2024-10-21'

data['timestamp'] = pd.to_datetime(data['timestamp'])

# ตรวจสอบว่า timestamp มี timezone หรือไม่
if not pd.api.types.is_datetime64tz_dtype(data['timestamp']):
    # ถ้าไม่มี timezone ให้กำหนดเป็น UTC แล้วแปลงเป็น US/Eastern
    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
else:
    # ถ้ามี timezone แล้ว แปลงเป็น US/Eastern
    data['timestamp'] = data['timestamp'].dt.tz_convert('US/Eastern')

# เรียงลำดับข้อมูลตามเวลาและรีเซ็ตดัชนี
data = data.sort_values(by='timestamp').reset_index(drop=True)

print(f'calculate_indicators doing ...........')
data = calculate_indicators(data)
print(f'calculate_indicators Done #########')
print(f'calculate_scores doing ...........')
data = calculate_scores(data)
print(f'calculate_scores Done #########')
print(f'generate_signals doing ...........')
data = generate_signals(data)
print(f'generate_signals Done #########')

num_buy_signals = data['Buy_Signal'].sum()
num_sell_signals = data['Sell_Signal'].sum()
print(f"Number of Buy Signals: {num_buy_signals}")
print(f"Number of Sell Signals: {num_sell_signals}")

print(f'create_labels doing ...........')
labeled_data = create_labels(data)
print(f'create_labels Done #########')
num_labels = labeled_data['Label'].value_counts()
print(f"Number of Labels:\n{num_labels}")


features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal', 
            'Envelope_Upper', 'Envelope_Lower', 'total_percent', 'total_sell_percent']

X = labeled_data[features]
y = labeled_data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                            cv=5, n_jobs=-1, verbose=2, scoring='f1')

grid_search.fit(X_train_scaled, y_train)
print("Best Parameters:")
print(grid_search.best_params_)
print("###########################")

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test_scaled)

print(f"Number of Buy Signals: {num_buy_signals}")
print(f"Number of Sell Signals: {num_sell_signals}")
print("Confusion Matrix (Best Model):")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best))


joblib.dump(best_model, 'best_model_xgb.pkl')
print("Model saved to: best_model_xgb.pkl")


joblib.dump(scaler, 'scaler_xgb.pkl')
print("Scaler saved to: scaler_xgb.pkl")


