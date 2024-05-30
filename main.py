import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, LabelEncoder
# from sklearn import metrics
from xailib.data_loaders.dataframe_loader import prepare_dataframe

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.compose import make_column_selector as selector

from lore_sa.dataset import TabularDataset
from lore_sa.surrogate import DecisionTreeSurrogate


def load_data_from_csv():
    df = pd.read_csv("../../datasets/Final_data.csv")

    df = df.loc[:, ['Class_label', 'Week5_Covid', 'Week4_Covid', 'Week3_Covid', 'Week5_Mobility', 'Week4_Mobility',
                    'Week3_Mobility', 'Week2_Mobility', 'Days_passed', 'Duration']]

    mask = df.drop(["Class_label"], axis=1).isna().all(axis=1) & df['Class_label'].notna()
    result_df = df[~mask]
    # df = df.dropna(how = 'all')
    result_df = result_df.fillna("NONE")

    return result_df

def create_and_train_model(result_df):
    y = result_df["Class_label"].values
    X_feat = result_df.loc[:, 'Week5_Covid':'Duration']
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(X_feat)
    categorical_columns = categorical_columns_selector(X_feat)
    covid_categories = ['NONE', 'c1', 'c2', 'c3', 'c4']
    mobility_categories = ['NONE', 'm1', 'm2', 'm3', 'm4']
    enc = OrdinalEncoder(
        categories=[covid_categories, covid_categories, covid_categories, mobility_categories, mobility_categories,
                    mobility_categories, mobility_categories])
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("ordinal-encoder", enc, categorical_columns),
            ("standard_scaler", numerical_preprocessor, numerical_columns),
        ]
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    model = make_pipeline(preprocessor, clf)

    data_train, data_test, target_train, target_test = train_test_split(
        X_feat, y, random_state=0
    )
    _ = model.fit(data_train, target_train)

    y_predict = model.predict(data_test)

    print(model.score(data_test, target_test))

    return data_train, clf

def generate_data_from_instance():
    res = load_data_from_csv()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_data_from_instance()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
