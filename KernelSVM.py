import matplotlib.pyplot as pyplot
import numpy
from sklearn.svm import SVC
import Plot

# ランダムなノイズを含んだXORデータセットを作成する

# 乱数種を指定
numpy.random.seed(0)

# 標準正規分布に従う200行2列の行列を生成
x_xor = numpy.random.randn(200, 2)

# 2つの引数に対して排他的論理和を作成
y_xor = numpy.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)

# 排他的論理和の値が正なら1, 負なら-1を作成
y_xor = numpy.where(y_xor, 1, -1)

# RBFカーネルによるSVMの学習を行う、gammaを大きくしすぎると過学習、Cを大きくすると誤分類に対してのぺナルティ大
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X=x_xor, y=y_xor)

# ----表示部-----
Plot.plot_decision_regions(x_combined_standard=x_xor, y_combined=y_xor, classifier=svm)

# # ラベルが1のxに対して青でプロット
# pyplot.scatter(x=x_xor[y_xor == 1, 0],
#                y=x_xor[y_xor == 1, 1],
#                c='blue',
#                marker='s',
#                label='1')
#
# # ラベルが-1のxに対してでプロット
# pyplot.scatter(x=x_xor[y_xor == -1, 0],
#                y=x_xor[y_xor == -1, 1],
#                c='red',
#                marker='x',
#                label='-1')

# 軸の範囲を指定
pyplot.xlim([-3, 3])
pyplot.ylim([-3, 3])

pyplot.legend(loc='best')

pyplot.show()

