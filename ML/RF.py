import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

# 訓練隨機森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 預測
y_pred = model.predict(X_test_scaled)

# 計算預測誤差（例如 MAE）
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# 繪製重要特徵圖
feature_importances = model.feature_importances_
features = X.columns

# 將特徵重要性結果轉換為 DataFrame 以便視覺化
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

importances_sorted_idx = np.argsort(feature_importances)[::-1]  # 由大到小排序
top_n = 10  # 顯示前10個最重要的特徵
top_n_features = np.array(features)[importances_sorted_idx][:top_n]
top_n_importances = feature_importances[importances_sorted_idx][:top_n]

plt.figure(figsize=(12, 8))
plt.bar(top_n_features, top_n_importances)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()