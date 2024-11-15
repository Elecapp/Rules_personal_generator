import os

import joblib
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.manifold import TSNE

import umap.umap_ as umap
#import umap.plot

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


import random
import math
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.lore import TabularRandomGeneratorLore, Lore

from sklearn.metrics import pairwise_distances


class NewGen(NeighborhoodGenerator):
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr)
        self.neighborhood = None
        self.preprocess = bbox.bbox.named_steps.get('columntransformer')
        self.gen_data = None

    def perturb(self, x, num_instances):
        perturbed_arrays = []
        for _ in range(num_instances):
            perturbed_arr = x[:]

            for val in range(0, int(len(x))):
                perturbed_arr[0] = random.uniform(0, 17.93)
                perturbed_arr[1] = random.uniform(perturbed_arr[0], 20.12)  # always greater than minspeed
                perturbed_arr[2] = random.uniform(perturbed_arr[1], 20.75)  # always greater than speedQ1
                perturbed_arr[3] = random.uniform(perturbed_arr[2], 21.65)  # always greater than speedMedian
                perturbed_arr[4] = random.uniform(0, 2.24)
                perturbed_arr[5] = random.uniform(-0.24, 0.36)
                perturbed_arr[6] = random.uniform(-2.80, 1.77)
                perturbed_arr[7] = random.uniform(0.12, 282.26)
                perturbed_arr[8] = math.log10(random.uniform(math.exp(-3.05), perturbed_arr[
                    7]))  # inverse log transform, identify value less than max dist, then log transform again

            perturbed_arrays.append(perturbed_arr)
        self.gen_data = perturbed_arrays
        return perturbed_arrays


        x = [0.03, 15.51, 15.91, 16.52, 0.004, 0.28, 0.98, 29.28, -1.86]
        perturber = NewGen(bbox, data, encoder)
        n = 10
        perturbed_arrays = perturber.perturb(x=x, num_instances=n)
        for perturbed_arr in perturbed_arrays:
            print(perturbed_arr)


def load_data_from_csv():
    df = pd.read_csv("datasets/final_df_addedfeat.csv")
    return df

def create_and_train_model(df):

    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']
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
    return model

def new_lore(res,model):
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    res = res.loc[:, features]
    instance = res.values[5, : -1]
    #prediction = model.predict([instance])
    instance = res.iloc[10]
    bbox = sklearn_classifier_bbox.sklearnBBox(model)

    print("instance is:", instance)
    x = pd.DataFrame([instance], columns=features)
    x['class N'] = pd.cut(x['class N'], bins=6, labels=['1', '2', '3','4','5','6'])
    #print('model prediction is', model.predict([x]))
    data = TabularDataset(data=x, class_name="class N")
    #lore = TabularRandomGeneratorLore(bbox, x)
    encoder = ColumnTransformerEnc(data.descriptor)
    surrogate = DecisionTreeSurrogate()
    generator = NewGen(bbox, data, encoder)
    proba_lore = Lore(bbox, data, encoder, generator, surrogate)
    rule = proba_lore.explain((x.values.reshape(1, -1)))

    print(rule)
    print('----- ')
    print(rule['rule'])
    print('----- counterfactual')
    for cr in rule['counterfactuals']:
        print(cr)
        print('-----')



if __name__ == '__main__':
    res = load_data_from_csv()
    model = create_and_train_model(res)
    print(model)

    new_lore(res, model)