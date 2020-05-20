import pandas as pd
import seaborn as sns
import numpy as np
import shap

from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


class Interpret:
    """
    Class to interpret a blackbox model.
    """
    def __init__(self, model, x_train=None, y_train=None, x_test=None, y_test=None):
        self.model = model
        self.x_train = None
        self.y_train = None
        self.predictions = []
        self.features = []
        self.x_test = x_test
        self.y_test = y_test
        if x_train and y_train:
            self.fit(x_train=x_train, y_train=y_train)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model.fit(x_train, y_train)
        self.features = x_train.columns.values

    def predict(self, x_test=None, y_test=None):
        self.x_test = x_test
        self.y_test = y_test
        if not isinstance(self.x_test, pd.DataFrame):
            raise Exception("A test set must be given to make a prediction!")
        if not isinstance(self.y_test, pd.DataFrame):
            raise Exception("Ground truth must be given to make a prediction!")

        self.predictions = self.model.predict(x_test)
        return self.predictions

    def feature_importance(self):
        if not self.model:
            raise Exception("There are no predictions yet!")
        weights = PermutationImportance(self.model).fit(self.x_test.values, self.y_test.values)
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

    def shap_interpret(self):
        """
        Method to interpret with SHAP values. This method supports only RandomForest!!!
        :return:
        """
        se = shap.TreeExplainer(self.model)  # , feature_perturbation="interventional", model_output="raw"
        shap_values = se.shap_values(self.x_test)
        shap.summary_plot(shap_values[1], features=self.x_test)  # feature_names=self.features

    def lime(self, instance=None, html_file=False):
        explainer = LimeTabularExplainer(self.x_train.values, mode="classification", feature_names=self.x_train.columns,
                                         class_names=['false', 'true'], training_labels=self.y_train, discretize_continuous=True)
        if not instance:
            instance = np.random.randint(0, self.x_test.shape[0])
            print('Case:  ' + str(instance))
            print('Label: ' + str(self.y_test.iloc[instance]))

        exp = explainer.explain_instance(self.x_test.values[instance], self.model.predict_proba, num_features=13)
        print("Lime explanation: ")
        exp.as_pyplot_figure(label=1).show()
        if html_file:
            exp.save_to_file(str(instance) + "_" + str(self.y_test.iloc[instance]) + "_explain.html")
# EOF
