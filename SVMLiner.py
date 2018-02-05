from sklearn.svm import SVC
import Utility

# データの前処理を行う
x_train_standard, x_test_standard, y_train, y_test = Utility.preprocessing_iris()

# Cの値が小さくなると、誤分類に対して寛容になる
svm = SVC(kernel='linear', C=100, random_state=0)
svm.fit(x_train_standard, y_train)

# トレーニング済みのモデルを用いてテストデータの予測を行う
Utility.predict(x_test_standard=x_test_standard, y_test=y_test, classifier=svm)

# 可視化を行う
Utility.visualize_iris(x_train_standard=x_train_standard,
                       x_test_standard=x_test_standard,
                       y_train=y_train,
                       y_test=y_test,
                       classifier=svm)