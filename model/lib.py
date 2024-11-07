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
from polygon import RESTClient
from datetime import datetime, timedelta, time, date
from pandas.tseries.offsets import BDay
import os
from dotenv import load_dotenv 
from ta import trend, momentum