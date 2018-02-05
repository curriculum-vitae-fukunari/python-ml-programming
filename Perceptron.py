from sklearn.linear_model import Perceptron
import Utility

# データの前処理を行う
x_train_standard, x_test_standard, y_train, y_test = Utility.preprocessing_iris()

# エポック数40, 学習率0.1でパーセプトロンのインスタンスを生成
perceptron = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)

# トレーニングデータをモデルに適合させる
perceptron.fit(x_train_standard, y_train)

# トレーニング済みのモデルを用いてテストデータの予測を行う
Utility.predict(x_test_standard=x_test_standard, y_test=y_test, classifier=perceptron)

# 可視化を行う
Utility.visualize_iris(x_train_standard=x_train_standard,
                       x_test_standard=x_test_standard,
                       y_train=y_train,
                       y_test=y_test,
                       classifier=perceptron)