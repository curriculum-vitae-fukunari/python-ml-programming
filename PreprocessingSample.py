import pandas
from io import StringIO
from sklearn.preprocessing import Imputer

# 欠損データへの対処

# サンプルデータを作成
csv_data = '''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# サンプルデータを読み込む
data_frame = pandas.read_csv(StringIO(csv_data))

# 各特徴量の欠測値をカウント
data_frame.isnull().sum()

#---欠測値を含む行や列を削除する---

# 欠測値を含む行を削除
data_frame.dropna()

# 欠測値を含む列を削除
data_frame.dropna(axis=1)

# 全ての列がNaNの行を削除
data_frame.dropna(how='all')

# 非NaN値が4つ未満の行を削除
data_frame.dropna(thresh=4)

# 特定の列にNaNが含まれている行だけ削除
data_frame.dropna(subset=['C'])

#---欠測値を補完---

# 欠測値を補完するインスタンスを生成
# axis=1にすると行の平均値になる
# strategy='most_frequent'にすると最頻値で補完できる
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# データを適合
imputer =imputer.fit(data_frame)

# 補完を実行
imputed_data = imputer.transform(data_frame.values)
