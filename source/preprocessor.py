import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def process(datapath):
    dataset = pd.read_csv(datapath)
    print(dataset.head(5))

    print(dataset.shape)

    dataset_missing = dataset.columns[dataset.isnull().any()].tolist()

    print(dataset_missing)
    X = dataset[['interest_rate',
                 'unpaid_principal_bal', 'loan_term',
                 'loan_to_value', 'number_of_borrowers',
                 'debt_to_income_ratio', 'borrower_credit_score',
                 'insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1',
                 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]  # input
    Y = dataset.iloc[0:25000, 28].values  # output
    # SPLITING THE DATA INTO TRAIN AND TEST

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)
    scale(X)

    return xtrain, xtest, ytrain, ytest


# SCALING THE DATA USING MIN-MAX SCALER

def scale(X):
    scaler = MinMaxScaler()
    numerical = ['interest_rate',
                 'unpaid_principal_bal', 'loan_term',
                 'loan_to_value', 'number_of_borrowers',
                 'debt_to_income_ratio', 'borrower_credit_score',
                 'insurance_percent', 'co-borrower_credit_score', 'm1',
                 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']

    features_minmax_transform = pd.DataFrame(data=X)
    features_minmax_transform[numerical] = scaler.fit_transform(X[numerical])
    print("features_minmax_transform:", features_minmax_transform)


if __name__ == "__main__":
    process()
