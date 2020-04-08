import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from source import preprocessor


class DecisionTree:
    datapath = None

    def __init__(self, data):
        self.datapath = data

    def generate(self):
        xtrain, xtest, ytrain, ytest = preprocessor.process(self.datapath)

        # DECISION TREE MODEL BUILDING:
        dct = DecisionTreeClassifier()
        dct.fit(xtrain, ytrain)
        tree_pred = dct.predict(xtest)

        a_dt, p_dt = metrics.accuracy_score(ytest, tree_pred), metrics.precision_score(ytest, tree_pred)
        print("Accuracy_decision_tree:", a_dt)
        print("Precision for DT:",p_dt)

        # CONFUSION MATRIX FOR DT
        confusion_matrix = metrics.confusion_matrix(ytest, tree_pred)
        print(confusion_matrix)
        # VISUALISATION OF DT:
        class_names = [0, 1]  # name of the classes
        fig, ax = plt.subplots()
        tick_set = np.arange(len(class_names))
        # plt.xticks(tick_set, class_names)
        # plt.yticks(tick_set, class_names)
        # CREATE A HEATMAP
        plt.figure(figsize=(6, 5))
        sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cbar_kws={'orientation': 'horizontal'}, cmap='Reds',
                    fmt='d')
        # plt.show()
        ax.xaxis.set_label_position("bottom")
        # plt.tight_layout()
        # plt.title("Exampleset")
        # plt.ylabel("Actual label")
        # plt.xlabel("predicted label")
        # ROC CURVE FOR DT:
        tree_pred_proba = dct.predict_proba(xtest)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(ytest, tree_pred_proba)
        # auc = metrics.roc_auc_score(ytest, tree_pred_proba)
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # plt.legend(loc=4)
        # plt.show()

        return "Accuracy_decision_tree:  " + str(a_dt) + "    Precision for DT:  " + str(p_dt)


if __name__ == "__main__":
    DecisionTree("../resources/Book2_1.csv").generate()
