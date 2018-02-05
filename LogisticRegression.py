from sklearn.linear_model import LogisticRegression
import Utility

# データの前処理を行う
x_train_standard, x_test_standard, y_train, y_test = Utility.preprocessing_iris()

# l2正則化、C値(1/λ)=1000でロジスティック回帰のインスタンスを作成
logistic_regression = LogisticRegression(penalty='l2', C=10000, random_state=0)

# 標準化したトレーニングデータとクラスラベルよりロジスティック回帰の学習を行う
logistic_regression.fit(X=x_train_standard, y=y_train)

# トレーニング済みのモデルを用いてテストデータの予測を行う
Utility.predict(x_test_standard=x_test_standard, y_test=y_test, classifier=logistic_regression)

# 可視化を行う
Utility.visualize_iris(x_train_standard=x_train_standard,
                       x_test_standard=x_test_standard,
                       y_train=y_train,
                       y_test=y_test,
                       classifier=logistic_regression)


