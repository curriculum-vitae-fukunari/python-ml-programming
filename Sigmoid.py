import matplotlib.pyplot as pyplot
import numpy

# シグモイド関数を定義
def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

# 0.1間隔で-7以上7未満のデータを作成
z = numpy.arange(start=-7.0, stop=7.0, step=0.1)

# 生成したデータでシグモイド関数を作成
phi_z = sigmoid(z)

# 元のデータとシグモイド関数の出力をプロット
pyplot.plot(z, phi_z)

# 垂直線を追加
pyplot.axvline(x=0.0, color='k')

# y軸の上限、下限を設定
pyplot.ylim(-0.1, 1.1)

# y軸の目盛りを設定
pyplot.yticks([0.0, 0.5, 1.0])

# Axesクラスのオブジェクトを作成
axes = pyplot.gca()

# 設定したy軸の目盛りに合わせて水平グリッド線を作成
axes.yaxis.grid(True)

# ラベルを設定
pyplot.xlabel('z')
pyplot.ylabel('$\phi (z)$')

# グラフを表示
pyplot.show()