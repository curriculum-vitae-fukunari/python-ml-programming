from sklearn.preprocessing import LabelEncoder
import pandas
import numpy

# サンプルデータを作成(Tシャツの色・サイズ・価格・クラスラベル)
data_frame = pandas.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])

data_frame.columns = ['color', 'size', 'price', 'classlabel']

print(data_frame)

# Tシャツのサイズと整数を対応させる辞書を生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
inv_size_mapping = {value: key for key, value in size_mapping.items()}

# Tシャツのサイズを整数に変換
data_frame['size'] = data_frame['size'].map(size_mapping)

# クラスラベルと整数を対応させる辞書を作成
class_mapping = {label: index for index, label in enumerate(numpy.unique(data_frame['classlabel']))}
inv_class_mapping = {value: key for key, value in class_mapping.items()}

# クラスラベルを整数に変換
data_frame['classlabel'] = data_frame['classlabel'].map(class_mapping)

# -------ラベルエンコーダーを用いる方法-------
class_label_encoder = LabelEncoder()
y = class_label_encoder.fit_transform(data_frame['classlabel'].values)

print(y)


# -------名義特徴量にone-hot-encodingを施す------