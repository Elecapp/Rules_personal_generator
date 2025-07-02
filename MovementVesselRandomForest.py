import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector

import vessels_utils

df = pd.read_csv("final_df_addedfeat.csv")

#features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'Log10Curvature',
#            'DistStartTrendAngle', 'Log10DistStartTrendDevAmplitude', 'MaxDistPort', 'Log10MinDistPort']
features = vessels_utils.vessels_features
label = "class N"

X_feat = df.loc[:, features]
y = df[label].values

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(X_feat)
categorical_columns = categorical_columns_selector(X_feat)

enc = OrdinalEncoder()
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