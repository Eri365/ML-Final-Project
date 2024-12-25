import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取原始資料檔案
file_path = 'weatherAUS.csv'  # 替換為你的資料檔案名稱
output_dir = 'Data'  # 輸出資料夾名稱

# 確保輸出資料夾存在
os.makedirs(output_dir, exist_ok=True)

# 讀取資料
data = pd.read_csv(file_path)

columns_to_drop = set() # 透過係數矩陣要捨去的欄位
threshold = 0.95    # 門檻值

# 根據 Location 分組並分別儲存成 CSV
for location, group in data.groupby('Location'):
    output_file = os.path.join(output_dir, f'{location}.csv')

    group = group.dropna() # 捨棄代有空資料的元素
    
    # 計算剩餘幾列資料，過少則捨棄該資料集
    if len(group) < 3000 * 0.95:
        continue
    group = group.drop(columns=['Location', 'RainToday', 'RainTomorrow'])    # 捨棄地區欄位
    
    group['NextDayRainfall'] = group['Rainfall'].shift(-1)  # 新增隔日降雨量欄位
    group = group.dropna() # 捨棄最後一欄資料 (沒有隔日降雨量)

    group = pd.get_dummies(group, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm']) # one-hot encoding

    df_without_date = group.drop(columns=['Date'])
    correlation_matrix = df_without_date.corr()
    # correlation_matrix = correlation_matrix.round(3)
    # correlation_matrix.to_csv('Pearson.csv', index=False)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                columns_to_drop.add(colname)

    group = group.drop(columns=columns_to_drop)

    group.to_csv(output_file, index=False)  # 寫檔
    
    print(f'Saved: {output_file}')