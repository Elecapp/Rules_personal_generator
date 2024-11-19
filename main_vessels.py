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

    def generate(self, x, num_instances: int, descriptor: dict, encoder):
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
        self.neighborhood = encoder.encode(perturbed_arrays)

        return self.neighborhood

def load_data_from_csv():
    df = pd.read_csv("datasets/final_df_addedfeat.csv")
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    df['class N'] = df['class N'].astype('str')

    return df[features]

def create_and_train_model(df):

    label = 'class N'
    features = df.columns[:-1]

    X_feat = df[features]
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
            ("standard_scaler", numerical_preprocessor, list(range(0,9))),
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


def new_lore(data, bb):
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    data = data.loc[:, features]
    #instance = data.values[5, : -1]
    #prediction = model.predict([instance])
    instance = data.iloc[10, :-1].values
    ds = TabularDataset(data=data, class_name='class N', categorial_columns=['class N'])
    bbox = sklearn_classifier_bbox.sklearnBBox(bb)
    print(bb.predict([instance]))

    print("instance is:", instance)
    x = instance
    #print('model prediction is', model.predict([x]))
    #lore = TabularRandomGeneratorLore(bbox, x)
    encoder = ColumnTransformerEnc(ds.descriptor)
    surrogate = DecisionTreeSurrogate()
    generator = NewGen(bbox, ds, encoder)
    proba_lore = Lore(bbox, ds, encoder, generator, surrogate)
    rule = proba_lore.explain(x)

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
    label = 'class N'
    features = res.columns[:-1]
    instance = res[features].loc[10].values
    print('prediction', model.predict([instance]))

    new_lore(res, model)