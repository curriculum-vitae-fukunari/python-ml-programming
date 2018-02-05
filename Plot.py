from matplotlib.colors import ListedColormap
import matplotlib.pyplot as pyplot
import numpy

# 決定境界のプロットを行う
def plot_decision_regions(x_combined_standard, y_combined, classifier, test_index=None, resolution=0.02):

    # マーカーとカラーを定義し、色のリストからカラーマップの作成を行う
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_map = ListedColormap(colors[:len(numpy.unique(y_combined))])

    # 決定領域のプロット

    # 2つの特徴量の最小値と最大値を求める
    x1_min = x_combined_standard[:, 0].min() - 1
    x1_max = x_combined_standard[:, 0].max() + 1
    x2_min = x_combined_standard[:, 1].min() - 1
    x2_max = x_combined_standard[:, 1].max() + 1

    # 特徴ベクトルを用いてグリッド配列xx1とxx2のペアを作成
    xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution), numpy.arange(x2_min, x2_max, resolution))

    # 各特徴量を一次元配列に変換して予測を実行
    z = classifier.predict(numpy.array([numpy.ravel(xx1), numpy.ravel(xx2)]).T)

    # 予測結果を元のグリッドポイントのデータサイズに変換
    z = z.reshape(xx1.shape)

    # グリッドポイントの等高線のプロット
    pyplot.contourf(xx1, xx2, z, cmap=color_map, alpha=0.4)

    # 軸の範囲を設定
    pyplot.xlim(xx1.min(), xx1.max())
    pyplot.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for index, class_ in enumerate(numpy.unique(y_combined)):
        pyplot.scatter(x=x_combined_standard[y_combined == class_, 0],
                       y=x_combined_standard[y_combined == class_, 1],
                       c=color_map(index),
                       alpha=0.8,
                       marker=markers[index],
                       s=10,
                       label=class_)

    # テストサンプルは目立たせるために点を○で表示
    if test_index:
        x_test = x_combined_standard[test_index, :]

        pyplot.scatter(x=x_test[:, 0],
                       y=x_test[:, 1],
                       alpha=0.2,
                       linewidths=1,
                       marker="o",
                       s=50,
                       label='test set')

