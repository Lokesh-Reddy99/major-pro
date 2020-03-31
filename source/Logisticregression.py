import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from source import preprocessor


class LogisticRegressionClass:
    datapath = None

    def __init__(self, data):
        self.datapath = data

    def generate(self):
        xtrain, xtest, ytrain, ytest = preprocessor.process(self.datapath)
        # TRAINING LOGISTIC REGRESSION MODEL
        logreg = LogisticRegression()
        logreg.fit(xtrain, ytrain)
        print(xtrain)
        y_pred = logreg.predict(xtest)  # CONFUSION MATRIX
        confusion_matrix = metrics.confusion_matrix(ytest, y_pred)
        print(confusion_matrix)

        # VISUALISATION
        class_names = [0, 1]  # name of the classes
        fig, ax = plt.subplots()
        tick_set = np.arange(len(class_names))
        plt.xticks(tick_set, class_names)
        plt.yticks(tick_set, class_names)

        # CREATE A HEATMAP
        plt.figure(figsize=(6, 5))
        sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cbar_kws={'orientation': 'horizontal'}, cmap='YlGnBu',
                    fmt='d')
        plt.show()
        ax.xaxis.set_label_position("bottom")
        plt.tight_layout()
        plt.title("Exampleset")
        plt.ylabel("Actual label")
        plt.xlabel("predicted label")

        # CONFUSION MATRIX EVALUATION METRICS
        print("Accuracy_LR:", metrics.accuracy_score(ytest, y_pred))
        print("Precision_LR:", metrics.precision_score(ytest, y_pred))

        # ROC CURVE
        y_pred_proba = logreg.predict_proba(xtest)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(ytest, y_pred_proba)
        auc = metrics.roc_auc_score(ytest, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()


if __name__ == "__main__":
    LogisticRegression("../resources/Book2_1.csv").generate()
