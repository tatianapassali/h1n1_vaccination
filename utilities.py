
from sklearn.metrics import accuracy_score, f1_score


def prediction_evaluation(prediction, y_test):
    print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))
    print('Test F1 score is {}'.format(f1_score(y_test, prediction)))
