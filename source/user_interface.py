from tkinter import *
from tkinter import filedialog
from source import Decisiontree, Logisticregression, RandomForest


class Application:

    def __init__(self):
        self.dataset_path = None
        self.logistic_regression, self.desicion_tree, self.random_forest = Logisticregression.LogisticRegressionClass(
            self.dataset_path), Decisiontree.DecisionTree(self.dataset_path), RandomForest.RandomForest(
            self.dataset_path)
        self.app = Tk()
        self.app.geometry("500x200")
        self.row_counter = 0
        self.filechosen = False

    def add_task(self, button_name, task_function, ):
        button = Button(self.app, text = button_name, command = task_function).grid(row = self.row_counter, column = 1)
        self.row_counter += 1
        return button

    def add_choose_file_button(self):
        row_num = self.row_counter
        def choose_file():
            self.dataset_path = filedialog.askopenfilename(initialdir="..", title="Select file", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
            Label(self.app, text="File : "+self.dataset_path).grid(row=row_num, column=1)
            self.logistic_regression, self.desicion_tree, self.random_forest = Logisticregression.LogisticRegressionClass(
                self.dataset_path), Decisiontree.DecisionTree(self.dataset_path), RandomForest.RandomForest(
                self.dataset_path)
            self.filechosen = True

        Button(self.app, text = "Choose Dataset", command = choose_file).grid(row = row_num, column = 0)
        self.row_counter += 1

    def taskloop(self):
        if self.filechosen:
            self.add_task("Generate Logistic Regression", self.logistic_regression.generate)
            self.add_task("Generate Decision Tree", self.desicion_tree.generate)
            self.add_task("Generate Random Forest", self.random_forest.generate)
        else:
            self.app.after(1000, self.taskloop)
    def main(self):
        self.add_choose_file_button()
        self.app.after(1000,self.taskloop)
        self.app.mainloop()

if __name__=="__main__":
    Application().main()
#
# def add_field(app, label_name):
#     global row_counter, column_counter
#     Label(app,text = label_name).grid(row = row_counter, column = 0)
#     e1 = Entry(app).grid(row = row_counter, column = 1)
#     row_counter += 1
#     return e1
#
# def add_multiple_fields(app, label_list):
#     field_list = []
#     for label in label_list:
#         field_list.append( add_field(app, label) )
#     return field_list