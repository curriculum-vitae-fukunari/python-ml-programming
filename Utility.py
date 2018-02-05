from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as pyplot
import numpy
import Plot

# アヤメのデータに関して前処理を行う
def preprocessing_iris():
    # Iris データセットをロード
    iris = datasets.load_iris()

    # 3, 4列目の特徴量を抽出
    x = iris.data[:, [2, 3]]

    # クラスラベルを取得
    y = iris.target

    # トレーニングデータとテストデータに分割
    # 全体の30%をテストデータとする
    # タプルで返している
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 特徴量のスケーリングが必要
    # StandardScalerを用いて特徴量を標準化する
    standard_scaler = StandardScaler()

    # トレーニングデータの平均と標準偏差を計算
    standard_scaler.fit(x_train)

    # 平均と標準偏差を用いて標準化
    x_train_standard = standard_scaler.transform(x_train)
    x_test_standard = standard_scaler.transform(x_test)

    return x_train_standard, x_test_standard, y_train, y_test

# 学習済みのモデルでテストデータを用いた予測を行い、正解率・予測結果などを表示する
def predict(x_test_standard, y_test, classifier):

    y_predict = classifier.predict(x_test_standard)

    # 誤分類のサンプルの個数を表示
    misclassified_sample_count = (y_test != y_predict).sum()
    print("misclassified: ", misclassified_sample_count)

    # 分類の正解率を表示
    accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
    print("accuracy: ", accuracy)

    # テストデータの先頭のサンプルがどのクラスラベルであるかの確率を予測する(現状ロジスティック回帰のみ)
    if type(classifier) == LogisticRegression:
        probability = classifier.predict_proba([x_test_standard[0, :]])
        print("probability: ", probability)

# 可視化を行う
def visualize_iris(x_train_standard, x_test_standard, y_train, y_test, classifier):

    x_combined_standard = numpy.vstack([x_train_standard, x_test_standard])
    y_combined = numpy.hstack([y_train, y_test])

    Plot.plot_decision_regions(x_combined_standard=x_combined_standard,
                               y_combined=y_combined,
                               classifier=classifier,
                               test_index=range(len(x_train_standard), len(x_combined_standard)))

    # 軸のラベルを設定
    pyplot.xlabel("petal length(standardized)")
    pyplot.ylabel("petal width(standardized)")

    # 凡例の設定
    pyplot.legend(loc='upper left')

    pyplot.show()