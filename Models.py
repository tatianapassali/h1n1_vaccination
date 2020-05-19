import pandas as pd
import seaborn as sns
import shap
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt


class Interpret:
    """
    Class to interpret a blackbox model.
    """
    def __init__(self, model, x_train=None, y_train=None, x_test=None):
        self.model = model
        self.predictions = []
        self.features = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        if x_train and y_train:
            self.fit(x_train=x_train, y_train=y_train)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model.fit(x_train, y_train)
        self.features = x_train.columns

    def predict(self, x_test=None):
        self.x_test = x_test
        if not isinstance(self.x_test, pd.DataFrame):
            raise Exception("A test set must be given to make a prediction!")

        self.predictions = self.model.predict(x_test)
        return self.predictions

    def feature_importance(self, y_test):
        if not self.model:
            raise Exception("There are no predictions yet!")
        weights = PermutationImportance(self.model).fit(self.x_test.values, y_test.values)
        model_weights = pd.DataFrame({'Features': list(self.features), 'Importance': weights.feature_importances_})
        model_weights = model_weights.reindex(model_weights['Importance'].abs().sort_values(ascending=False).index)
        model_weights = model_weights[(model_weights["Importance"] != 0)]
        self.plot(model_weights)

    def plot(self, data):
        plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        sns.barplot(x="Importance", y="Features", data=data)
        # plt.title("Intercept (Bias): " + str(self.model.intercept_[0]), loc='right')
        plt.xticks(rotation=90)
        plt.show()

    def partial_dependence_plots(self, y_test):
        if not self.predictions:
            raise Exception("There are no predictions yet!")

        # TODO

    def shap_interpret(self, y_test):

        shap.initjs()
        se = shap.TreeExplainer(self.model)  # , feature_perturbation="interventional", model_output="raw"
        shap_values = se.shap_values(self.x_test)
        shap.summary_plot(shap_values[1], features=self.x_test, feature_names=self.features)

    def lime(self):
        pass

# EOF
