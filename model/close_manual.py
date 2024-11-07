import json
import pandas as pd
from datetime import datetime, timedelta
import requests 

def create_message(stockName, price, takeProfit, stopLoss, qty, side):
        message = [{'botId': str('robot.tgai_tana_000_test@techglobetrading.com'),
                'stockName': str(stockName),
                'timestamp': str(pd.Timestamp(datetime.now().strftime("%Y-%m-%d %H:%M:%S")).tz_localize("US/Eastern")),
                'price': float(price),
                'takeProfit': float(takeProfit),
                'stopLoss': float(stopLoss),
                'qty': float(qty),
                'strategy': str('TANA001'),
                'side': str(side),
                'type': str('market'),
                'broker': str('alpaca')}]
        
        url = 'https://robot.techglobetrading.com/api/createorder'
        
        for m in message:
            response = requests.post(url, data=m)
            if response.status_code == 200:
                print(" [/] Sent %r" % m)
            else:
                print(" [x] Failed to send %r" % m)
                print(" [x] Response: %s" % response.text)

        

# create_message('COIN', 201, 210, 200, 100, 'buy')
create_message('COIN', 201, 210, 200, 100, 'sell')
