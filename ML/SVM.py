import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

data = pd.read_csv('Data/Brisbane.csv')

# 捨去日期欄位
data = data.drop(columns=['Date'])

X = data.drop(['NextDayRainfall'], axis=1)  # 特徵欄位
y = data['NextDayRainfall']  # 目標欄位

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練模型
svr = SVR()
svr.fit(X_train, y_train)

# 預測
y_pred = svr.predict(X_test)

# 計算 MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# 繪製實際值與預測值
plt.figure(figsize=(14, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')  # 實際值
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')  # 預測值
plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('NextDayRainfall')
plt.show()
