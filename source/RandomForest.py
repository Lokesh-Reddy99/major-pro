import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from source import preprocessor


class RandomForest:
    datapath = None

    def __init__(self, data):
        self.datapath = data

    def generate(self):
        xtrain, xtest, ytrain, ytest = preprocessor.process(self.datapath)
        # IMPORT RANDOM FOREST
        rfc = RandomForestClassifier()
        rfc.fit(xtrain, ytrain)
        rfc_pred = rfc.predict(xtest)
        print("Accuracy_RFC:", metrics.accuracy_score(ytest, rfc_pred))
        print("Precision for RF:", metrics.precision_score(ytest, rfc_pred))

        # CONFUSION MATRIX FOR RF:
        confusion_matrix = metrics.confusion_matrix(ytest, rfc_pred)
        print(confusion_matrix)
        # VISUALISATION of RF

        class_names = ['interest_rate',
                       'unpaid_principal_bal', 'loan_term',
                       'loan_to_value', 'number_of_borrowers',
                       'debt_to_income_ratio', 'borrower_credit_score',
                       'insurance_percent', 'co-borrower_credit_score', 'm1',
                       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']  # name of the classes

        class_names_y = ["0", "1"]
        fig, ax = plt.subplots()

        tick_set = np.arange(len(class_names))
        plt.xticks(tick_set, class_names)
        plt.yticks(tick_set, )

        # CREATE A HEATMAP
        plt.figure(figsize=(6, 5))
        sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cbar_kws={'orientation': 'horizontal'}, cmap='RdPu',
                    fmt='g')
        plt.show()
        ax.xaxis.set_label_position("bottom")
        plt.tight_layout()
        plt.title("Exampleset")
        plt.ylabel("Actual label")
        plt.xlabel("predicted label")

        # ROC CURVE

        rfc_pred_proba = rfc.predict_proba(xtest)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(ytest, rfc_pred_proba)
        auc = metrics.roc_auc_score(ytest, rfc_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()


if __name__ == "__main__":
    RandomForest("../resources/Book2_1.csv").generate()
