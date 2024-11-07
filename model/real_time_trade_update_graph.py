# trade_bot.py

import asyncio
import json
import pandas as pd
import pytz
from datetime import datetime, timedelta
import aiohttp
import logging
from threading import Thread, Lock
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import requests
import joblib
import numpy as np
from dotenv import load_dotenv 
import os
from utils import calculate_indicators, calculate_scores_for_real_trade, generate_signals_for_real_trade, create_labels

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

API_KEY = os.getenv('API_KEY')  
SYMBOL = 'SMCI'  

CREATE_ORDER_URL = os.getenv('CREATE_ORDER_URL')
BOT_ID = os.getenv('BOT_ID')  

# please change BD

# calculate_scores
max_buy_score =600
max_sell_score =600

# generate_signals
buy_score =50
sell_score =50
capital =80000

# create_labels
profit_p = 0.5
loss_p = -0.1

# setting
initial_capital = capital
shares = 0
position = False
entry_price = 0.0
trailing_stop = False
stop_loss = 0.0

market_open_time = datetime.strptime('09:30:00', '%H:%M:%S').time()
market_close_time = datetime.strptime('15:45:00', '%H:%M:%S').time()

columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
           'RSI', 'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal',
           'Envelope_Upper', 'Envelope_Lower',
           'score_ema', 'score_macd', 'score_rsi30', 'score_envelope', 'score_bullCond',
           'sell_score_ema', 'sell_score_macd', 'sell_score_rsi70', 'sell_score_envelope', 'sell_score_bearCond',
           'score', 'sell_score', 'total_percent', 'total_sell_percent',
           'Buy_Signal', 'Sell_Signal', 'capital', 'equity', 'Label']  
data = pd.DataFrame(columns=columns)
data_lock = Lock()

try:
    model = joblib.load('best_model_xgb.pkl')
    scaler = joblib.load('scaler_xgb.pkl')
    logging.info("download modle scaler finished")
except Exception as e:
    logging.error(f"ไม่สามารถโหลดโมเดลหรือ scaler ได้: {e}")
    exit(1) 

async def send_order(session, stock_name, timestamp, price, take_profit, stop_loss, qty, side):
    logging.info(f"กำลังส่งคำสั่ง {side.upper()} | จำนวน: {qty}, ราคา: {price}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")

    message = [{'botId': str(BOT_ID),
                'stockName': str(stock_name),
                'timestamp': str(pd.Timestamp(datetime.now().strftime("%Y-%m-%d %H:%M:%S")).tz_localize("US/Eastern")),
                'price': float(price),
                'takeProfit': float(take_profit),
                'stopLoss': float(stop_loss),
                'qty': float(qty),
                'strategy': str('TANA001'),
                'side': str(side),
                'type': str('market'),
                'broker': str('alpaca')}]
    
    headers = {
        'Content-Type': 'application/json'
    }
    print("--------------------------------MM------------------------------------------")
    print(message)
    
    try:
        url = 'https://robot.techglobetrading.com/api/createorder'        
        for m in message:
            response = requests.post(url, data=m)
            if response.status_code == 200:
                print(" [/] Sent %r" % m)
            else:
                print(" [x] Failed to send %r" % m)
                print(" [x] Response: %s" % response.text)
#################################################################################################
    except Exception as e:
        logging.error(f"ข้อยกเว้นขณะส่งคำสั่ง {side}: {e}")

def create_dash_app():
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1(f"Real-Time Trading Bot for {SYMBOL}"),
        dcc.Graph(id='live-candlestick'),
        dcc.Interval(
            id='interval-component',
            interval=15000,  
            n_intervals=0
        )
    ])

    @app.callback(Output('live-candlestick', 'figure'),
                  [Input('interval-component', 'n_intervals')])
    def update_graph_live(n):
        with data_lock:
            if data.empty:
                logging.debug("DataFrame ว่างเปล่า ไม่มีข้อมูลให้ Plot")
                return go.Figure()

            logging.debug("เริ่มสร้าง Figure ใหม่ใน Dash Callback")

            recent_data = data.tail(100)

            candlestick = go.Candlestick(
                x=recent_data['timestamp'],
                open=recent_data['open'],
                high=recent_data['high'],
                low=recent_data['low'],
                close=recent_data['close'],
                name='Candlestick'
            )

            volume = go.Bar(
                x=recent_data['timestamp'],
                y=recent_data['volume'],
                marker_color='rgba(0,0,255,0.3)',
                name='Volume',
                yaxis='y2'
            )

            buy_signals = recent_data[recent_data['Buy_Signal']]
            buy_marker = go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['low'] * 0.99, 
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy Signal'
            )


            sell_signals = recent_data[recent_data['Sell_Signal']]
            sell_marker = go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['high'] * 1.01,  
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell Signal'
            )

            label_up = recent_data[recent_data['Label'] == 1]
            label_down = recent_data[recent_data['Label'] == 0]
            traces = [candlestick, volume, buy_marker, sell_marker]

            if not label_up.empty:
                label_up_marker = go.Scatter(
                    x=label_up['timestamp'],
                    y=label_up['low'] * 0.98,
                    mode='markers',
                    marker=dict(symbol='circle', color='blue', size=8),
                    name='Label Up (1)'
                )
                traces.append(label_up_marker)

            if not label_down.empty:
                label_down_marker = go.Scatter(
                    x=label_down['timestamp'],
                    y=label_down['high'] * 1.02,
                    mode='markers',
                    marker=dict(symbol='circle', color='orange', size=8),
                    name='Label Down (0)'
                )
                traces.append(label_down_marker)

            predicted_up = data[data['Predicted_Label'] == 1] if 'Predicted_Label' in data.columns else pd.DataFrame()
            predicted_down = data[data['Predicted_Label'] == 0] if 'Predicted_Label' in data.columns else pd.DataFrame()

            if not predicted_up.empty:
                predicted_up_marker = go.Scatter(
                    x=predicted_up['timestamp'],
                    y=predicted_up['low'] * 0.97,
                    mode='markers',
                    marker=dict(symbol='diamond', color='purple', size=10),
                    name='Predicted Label Up (1)'
                )
                traces.append(predicted_up_marker)

            if not predicted_down.empty:
                predicted_down_marker = go.Scatter(
                    x=predicted_down['timestamp'],
                    y=predicted_down['high'] * 1.03,
                    mode='markers',
                    marker=dict(symbol='diamond', color='magenta', size=10),
                    name='Predicted Label Down (0)'
                )
                traces.append(predicted_down_marker)

            fig = go.Figure(data=traces)

            fig.update_layout(
                title=f"Real-Time Trading for {SYMBOL}",
                xaxis=dict(title='Time'),
                yaxis=dict(title='Price'),
                yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                legend=dict(x=0, y=1.2, orientation='h'),
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='x unified'
            )

            logging.debug("build Figure in Dash Callback")
            return fig

    return app

async def get_historical_data():
    global data  
    logging.info("start pulling data")
    async with aiohttp.ClientSession() as session:
        try:
            to_datetime = datetime.now(pytz.timezone('US/Eastern'))
            from_datetime = to_datetime - timedelta(days=2)
            to_date = to_datetime.strftime('%Y-%m-%d')
            from_date = from_datetime.strftime('%Y-%m-%d')

            url = f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/15/second/{from_date}/{to_date}?apiKey={API_KEY}&limit=50000"
            logging.info(f"pulling data from URL.....")

            async with session.get(url) as response: # เป็น Get
                if response.status == 200:
                    historical_data = await response.json()
                    if 'results' in historical_data:
                        logging.info(f"found {len(historical_data['results'])} แถวของข้อมูลย้อนหลัง")
                        with data_lock:
                            for result in historical_data['results']:
                                try:
                                    timestamp = datetime.fromtimestamp(result['t'] / 1000, pytz.UTC).astimezone(pytz.timezone('US/Eastern'))
                                    new_row = {
                                        'timestamp': timestamp,
                                        'open': float(result['o']),
                                        'high': float(result['h']),
                                        'low': float(result['l']),
                                        'close': float(result['c']),
                                        'volume': float(result['v']),
                                        'RSI': np.nan,  
                                        'EMA_Fast': np.nan,
                                        'EMA_Slow': np.nan,
                                        'MACD': np.nan,
                                        'MACD_Signal': np.nan,
                                        'Envelope_Upper': np.nan,
                                        'Envelope_Lower': np.nan,
                                        'score_ema': 0,
                                        'score_macd': 0,
                                        'score_rsi30': 0,
                                        'score_envelope': 0,
                                        'score_bullCond': 0,
                                        'sell_score_ema': 0,
                                        'sell_score_macd': 0,
                                        'sell_score_rsi70': 0,
                                        'sell_score_envelope': 0,
                                        'sell_score_bearCond': 0,
                                        'score': 0,
                                        'sell_score': 0,
                                        'total_percent': 0,
                                        'total_sell_percent': 0,
                                        'Buy_Signal': False,
                                        'Sell_Signal': False,
                                        'capital': capital,
                                        'equity': capital,
                                        'Label': pd.NA  # เพิ่ม Label
                                    }
                                    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                                    logging.debug(f"add new row data: {new_row}")
                                except Exception as e:
                                    logging.error(f"error new row: {e}")

    
                        data.sort_values('timestamp', inplace=True)
                        data.reset_index(drop=True, inplace=True)

                        logging.info("กำลังอัปเดตตัวชี้วัดและคะแนนหลังจากดึงข้อมูลย้อนหลัง")
                        calculate_indicators(data)
                        calculate_scores_for_real_trade(data, max_buy_score, max_sell_score)
                        generate_signals_for_real_trade(data, buy_score, sell_score, capital)  
                        logging.info("เสร็จสิ้นการอัปเดตตัวชี้วัดและคะแนนหลังจากดึงข้อมูลย้อนหลัง")
                    else:
                        logging.error("ไม่มีข้อมูล 'results' ในการตอบกลับ")
                else:
                    logging.error(f"ข้อผิดพลาดในการดึงข้อมูลย้อนหลัง: {response.status} - {await response.text()}")
        except Exception as e:
            logging.error(f"ข้อผิดพลาดในการดึงข้อมูลย้อนหลัง: {e}")

async def fetch_real_time_trade(session):
    url = f"https://api.polygon.io/v2/last/trade/{SYMBOL}?apiKey={API_KEY}"
    logging.debug(f"get real time data from URL........")
    try:
        async with session.get(url) as response:
            if response.status == 200:
                trade = await response.json()
                logging.debug(f"get real time data success: {trade}")
                return trade
            else:
                logging.error(f"ข้อผิดพลาดในการดึงข้อมูลการเทรดเรียลไทม์: {response.status} - {await response.text()}")
                return None
    except Exception as e:
        logging.error(f"ข้อผิดพลาดในการเชื่อมต่อกับ API: {e}")
        return None

async def handle_trading(session, current_close, in_market_hours, timestamp):
    global capital, shares, position, entry_price, trailing_stop, stop_loss

    logging.info(f"เริ่มจัดการการซื้อขาย ณ เวลา {timestamp}")

    if position:
        logging.debug("มีตำแหน่งเปิดอยู่ กำลังตรวจสอบการตัดขาดทุนและการรับกำไร")

        if current_close <= entry_price * 0.99:
            logging.info(f"ราคาลดลงเกิน 1% จากราคาที่ซื้อ ({entry_price})")

            with data_lock:
                data.at[data.index[-1], 'Sell_Signal'] = True
            capital += shares * current_close
            shares = 0
            position = False
            entry_price = 0.0
            trailing_stop = False  
            logging.info(f"ขายทั้งหมดเพื่อรับประกันการตัดขาดทุน | เงินทุนปัจจุบัน: {capital}")
        else:
            if not trailing_stop and current_close >= entry_price * 1.005:
                logging.info(f"กำไรเกิน 0.5% จากราคาที่ซื้อ ({entry_price}) เริ่มใช้ trailing stop")
                trailing_stop = True
                stop_loss = entry_price  

            if trailing_stop:
                if current_close <= stop_loss:
                    logging.info(f"ราคาลดลงถึงจุดตัดขาดทุน ({stop_loss})")

                    with data_lock:
                        data.at[data.index[-1], 'Sell_Signal'] = True
                    capital += shares * current_close
                    shares = 0
                    position = False
                    entry_price = 0.0
                    trailing_stop = False  
                    logging.info(f"ขายทั้งหมดเพื่อรับประกันการตัดขาดทุน | เงินทุนปัจจุบัน: {capital}")

            if current_close >= entry_price * 1.01:
                logging.info(f"กำไรเกิน 1% จากราคาที่ซื้อ ({entry_price})")

                with data_lock:
                    data.at[data.index[-1], 'Sell_Signal'] = True
                capital += shares * current_close
                shares = 0
                position = False
                entry_price = 0.0
                trailing_stop = False 
                logging.info(f"ขายทั้งหมดเพื่อรับกำไร | เงินทุนปัจจุบัน: {capital}")


    buy_condition = False
    sell_condition = False
    shares_to_buy = 0
    with data_lock:
        if not data.empty:
            buy_condition = (
                data['total_percent'].iloc[-1] > 25 and
                not position and
                in_market_hours 
            )
            sell_condition = (
                data['total_sell_percent'].iloc[-1] > 25 and  
                position and
                in_market_hours
            )

    label = None

    if buy_condition:
        with data_lock:

            feature_columns = ['RSI', 'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal', 
                               'Envelope_Upper', 'Envelope_Lower', 'total_percent', 'total_sell_percent']
            latest_data = data.iloc[-1][feature_columns].values.reshape(1, -1)

            try:
                latest_data = latest_data.astype(float)
            except ValueError as e:
                logging.error(f"การแปลงข้อมูลเป็น float ไม่สำเร็จ: {e}")
                latest_data = np.nan * latest_data  

            latest_data = np.where(np.isnan(latest_data), data[feature_columns].mean().values, latest_data)

            logging.debug(f"Latest data for prediction: {latest_data}")
            print("Latest data for prediction:", latest_data)

            if not np.isnan(latest_data).any():
                X_scaled = scaler.transform(latest_data)
                try:
                    label = model.predict(X_scaled)[0]
                    data.at[data.index[-1], 'Label'] = label
                    logging.debug(f"ทำนาย Label สำหรับข้อมูลล่าสุด: {label}")
                except Exception as e:
                    logging.error(f"ข้อผิดพลาดในการทำนาย Label: {e}")
            else:
                logging.warning("ยังมีค่า NaN ในฟีเจอร์ ไม่สามารถทำนาย Label ได้")

    final_buy_condition = buy_condition and (label == 1)

    if final_buy_condition:
        logging.info("เงื่อนไขการซื้อเป็นจริง และ Label จากโมเดลคือ 1")

        shares_to_buy = int(capital // current_close) 
        if shares_to_buy > 0:
            logging.info(f"ซื้อ {shares_to_buy} หุ้น ณ ราคา {current_close}")
            with data_lock:
                data.at[data.index[-1], 'Buy_Signal'] = True
            capital -= shares_to_buy * current_close 
            shares += shares_to_buy
            position = True
            entry_price = current_close  
            trailing_stop = False  
            logging.info(f"ซื้อเรียบร้อย | จำนวนหุ้น: {shares} | เงินทุนปัจจุบัน: {capital}")

            take_profit = current_close * 1.01  
            stop_loss = current_close * 0.99   

            await send_order(
                session=session,
                stock_name=SYMBOL,
                timestamp=timestamp,
                price=current_close,
                take_profit=take_profit,
                stop_loss=stop_loss,
                qty=shares_to_buy,
                side='buy'
            )
    final_sell_condition = sell_condition
    final_sell_condition = sell_condition
    if sell_condition and shares > 0:
        logging.info("sell condition is true")

        take_profit = 1
        stop_loss = 1
        qty = shares  

        await send_order(
            session=session,
            stock_name=SYMBOL,
            timestamp=timestamp,
            price=current_close,
            take_profit=take_profit,
            stop_loss=stop_loss,
            qty=qty,
            side='sell'
        )
        logging.info(f"Sell all  {shares} at price {current_close}")
        with data_lock:
            data.at[data.index[-1], 'Sell_Signal'] = True
        capital += shares * current_close
        shares = 0
        position = False
        entry_price = 0.0
        trailing_stop = False  
        
        await send_order(
            session=session,
            stock_name=SYMBOL,
            timestamp=timestamp,
            price=current_close,
            take_profit=take_profit,
            stop_loss=stop_loss,
            qty=qty,
            side='sell'
        )
        
        print(f"sell all done")

    if not in_market_hours:
        logging.info("Market is closing please sell all")
        if position and shares > 0:
            with data_lock:
                data.at[data.index[-1], 'Sell_Signal'] = True
            
              
            capital += shares * current_close
            shares = 0
            position = False
            entry_price = 0.0
            trailing_stop = False  
            await send_order(
                session=session,
                stock_name=SYMBOL,
                timestamp=timestamp,
                price=current_close,
                take_profit=take_profit,
                stop_loss=stop_loss,
                qty=qty,
                side='sell'
            )
            logging.info(f"Sell all at market close | cost now: {capital}")

    equity = capital + shares * current_close
    with data_lock:
        data.at[data.index[-1], 'capital'] = capital
        data.at[data.index[-1], 'equity'] = equity
    logging.debug(f"Cal equity: {equity}")

async def poll_rest_api():
    global data  
    logging.info("เริ่มรันการดึงข้อมูลเรียลไทม์")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                trade = await fetch_real_time_trade(session)
                logging.debug(f"ค่าของ trade หลังจาก fetch_real_time_trade: {trade}")
                if trade and 'results' in trade:
                    last_trade = trade['results']
                    timestamp = datetime.fromtimestamp(last_trade['t'] / 1e9, pytz.UTC).astimezone(pytz.timezone('US/Eastern'))
                    price = last_trade.get('p', 0)  
                    volume = last_trade.get('q', 0)  

                    logging.debug(f"ข้อมูลการเทรดล่าสุด: เวลา={timestamp}, ราคา={price}, ปริมาณ={volume}")

                    with data_lock:

                        if not data.empty and timestamp.replace(second=0, microsecond=0) == data['timestamp'].iloc[-1].replace(second=0, microsecond=0):
                            logging.debug("อัปเดต candle ล่าสุด")
                            data.at[len(data)-1, 'high'] = max(data.at[len(data)-1, 'high'], price)
                            data.at[len(data)-1, 'low'] = min(data.at[len(data)-1, 'low'], price)
                            data.at[len(data)-1, 'close'] = price
                            data.at[len(data)-1, 'volume'] += volume
                        else:
                            logging.debug("สร้าง candle ใหม่")
                            new_row = {
                                'timestamp': timestamp,
                                'open': float(price),   
                                'high': float(price),
                                'low': float(price),
                                'close': float(price),
                                'volume': float(volume),
                                'RSI': np.nan,  
                                'EMA_Fast': np.nan,
                                'EMA_Slow': np.nan,
                                'MACD': np.nan,
                                'MACD_Signal': np.nan,
                                'Envelope_Upper': np.nan,
                                'Envelope_Lower': np.nan,
                                'score_ema': 0,
                                'score_macd': 0,
                                'score_rsi30': 0,
                                'score_envelope': 0,
                                'score_bullCond': 0,
                                'sell_score_ema': 0,
                                'sell_score_macd': 0,
                                'sell_score_rsi70': 0,
                                'sell_score_envelope': 0,
                                'sell_score_bearCond': 0,
                                'score': 0,
                                'sell_score': 0,
                                'total_percent': 0,
                                'total_sell_percent': 0,
                                'Buy_Signal': False,
                                'Sell_Signal': False,
                                'capital': capital,
                                'equity': capital,
                                'Label': pd.NA  
                            }

                            feature_columns = ['RSI', 'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal', 
                                               'Envelope_Upper', 'Envelope_Lower', 'total_percent', 'total_sell_percent']
                            for col in feature_columns:
                                try:
                                    new_row[col] = float(new_row[col]) if not pd.isna(new_row[col]) else np.nan
                                except ValueError as e:
                                    logging.error(f"การแปลงคอลัมน์ {col} เป็น float ไม่สำเร็จ: {e}")
                                    new_row[col] = np.nan

                            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                            logging.debug(f"เพิ่ม candle ใหม่: {new_row}")

                    with data_lock:
                        calculate_indicators(data)
                        calculate_scores_for_real_trade(data, max_buy_score, max_sell_score)
                        generate_signals_for_real_trade(data, buy_score, sell_score, capital) 
                        logging.debug(f"ข้อมูล DataFrame ล่าสุด:\n{data.tail()}") 

                    await handle_trading(
                        session=session,
                        current_close=price,
                        in_market_hours=(market_open_time <= timestamp.time() <= market_close_time),
                        timestamp=timestamp
                    )

                else:
                    logging.warning("ไม่พบคีย์ 'results' ในข้อมูล trade หรือ trade เป็น None")

                await asyncio.sleep(15)
            except Exception as e:
                logging.error(f"เกิดข้อผิดพลาด: {e}")
                await asyncio.sleep(5) 

async def main_async():
    logging.info("Start Robot ...... ")
    await get_historical_data()
    await poll_rest_api()

def run_dash_app():
    app = create_dash_app()
    app.run_server(debug=False, use_reloader=False, port=8060)

def main():
    dash_thread = Thread(target=run_dash_app)
    dash_thread.start()
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        logging.info("หยุดการทำงานของโปรแกรมโดยผู้ใช้")
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดใน loop: {e}")
    finally:
        loop.stop()
        loop.close()
        logging.info("ปิด Event Loop")
        dash_thread.join()

if __name__ == "__main__":
    main()
