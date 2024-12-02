import os

import joblib
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn import tree
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

class GenerateDecisionTrees:
    def __init__(self, test_size=0.3, random_state=42):
        self.random_state = random_state
        self.test_size = test_size
        self.classifiers = {}

    def decision_trees(self,X_feat, y):
        self.classifiers = {}
        for target_class in np.unique(y):
            binary_y = (y == target_class).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X_feat, binary_y, random_state=self.random_state, test_size=self.test_size)
            clf = DecisionTreeClassifier(random_state=self.random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # print(f"Classification Report for class {target_class}:\n")
            # print(classification_report(y_test, y_pred))
            self.classifiers[target_class] = clf
        return self.classifiers



class NewGen(NeighborhoodGenerator):
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, classifiers: dict, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr)
        self.neighborhood = None
        self.preprocess = bbox.bbox.named_steps.get('columntransformer')
        self.gen_data = None
        self.classifiers = classifiers


    def generate(self, x, num_instances:int=1000, descriptor: dict=None, encoder=None, list=None):
        perturbed_arrays = []
        perturbed_arrays.append(x.copy())



        '''
        for _ in range(num_instances):
        # make a deep copy of x in the variable perturbed_arr
            perturbed_arr = x.copy()
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
            perturbed_arrays.append(perturbed_arr[:])
        '''
        for _ in range(num_instances):
            perturbed_x = self.perturbate(x)
            perturbed_arrays.append(perturbed_x)
        return perturbed_arrays

    def perturbate(self, instance):
        class_label = self.bbox.predict([instance])[0]
        chosen_dt = self.classifiers[class_label]

        # Get the decision path and leaf node for the instance
        decision_path = chosen_dt.decision_path(instance.reshape(1,-1))
        leaf_id = chosen_dt.apply(instance.reshape(1,-1))[0]

        feature = chosen_dt.tree_.feature
        threshold = chosen_dt.tree_.threshold
        # Extract influencing features based on the decision path
        node_indices = decision_path.indices  # Nodes visited along the path
        influencing_features = []

        for node in node_indices:
            if feature[node] != -2:  # -2 indicates a leaf node
                influencing_features.append(
                    (X_feat.columns[feature[node]], threshold[node], instance[feature[node]], instance[feature[node]] < threshold[node], node)
                )
        # extract the feature indices
        feature_indices = [f[4] for f in influencing_features]

        # make a deep copy of x in the variable perturbed_arr
        perturbed_arr = instance.copy()
        perturbed_arr[0] = random.uniform(0, 17.93)
        perturbed_arr[1] = random.uniform(perturbed_arr[0], 20.12)  # always greater than minspeed
        perturbed_arr[2] = random.uniform(perturbed_arr[1], 20.75)  # always greater than speedQ1
        perturbed_arr[3] = random.uniform(perturbed_arr[2], 21.65)  # always greater than speedMedian
        perturbed_arr[4] = random.uniform(0, 2.24)
        perturbed_arr[5] = random.uniform(-0.24, 0.36)
        perturbed_arr[6] = random.uniform(-2.80, 1.77)
        perturbed_arr[7] = random.uniform(0.12, 282.26)
        perturbed_arr[8] = math.log10(random.uniform(math.exp(-3.05), perturbed_arr[7]))  # inverse log transform, identify value less than max dist, then log transform again

        mask_indices = [perturbed_arr[i] if i in feature_indices else instance[i] for i, v in enumerate(instance)]

        return mask_indices







        '''
        in the class there are already the stored decision tree, 
        I retrieve the decision tree
        '''
        '''
        New alg description:
        - check the class label of x
        - find the corresponding decision tree for the class label
        - find the corresponding branch in the DT for the instance
        - generate the perturbations by altering feature values to 
          fit into other branches of the DT (threshold of same class vs different class) and ensure
          that the soft rules are maintained
        - Keep the feature that do not influence the class decision as constants.
        
        ---------------
        def create_and_train_model(df):
            return bbox
        #create class with the precalc decision tree
        def decision_trees(x)
            classifiers = {}
            for target_class in np.unique(y):
                binary_y = (y == target_class).astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X_feat, binary_y, test_size=0.3, random_state=42)
                clf = DecisionTreeClassifier(random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                #print(f"Classification Report for class {target_class}:\n")
                #print(classification_report(y_test, y_pred))
                classifiers[target_class] = clf
            return classifiers
        x = [instance]
        x_class= bbox.predict(x) #check class label for x
        chosen_dt = classifiers[x_class] # find the corresponding DT
        check_x = x.reshape(1, -1)
        # Step 1: Get the decision path
        node_indicator = clf.decision_path(check_x)

        # Step 2: Find the leaf node for the instance
        leaf_id = clf.apply(check_x)[0]

        # Step 3: Extract conditions along the path
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        #generate the perturbations by altering feature values to fit into other branches of the DT (threshold of same 
        class vs different class)
        #Keep the feature that do not influence the class decision as constants
        candidate_neigh = [x.perturb] # perturb everything but the irrelevant feature, leverage on the def generate
        check what is inside node indicator.
        for n in candidate_neigh: #ensure that the soft rules are maintained
            check if it fits the soft rules
            if not:
                modify accordingly
            proper_neigh   
        '''

        return self.neighborhood

def load_data_from_csv():
    class_names = {
        '1': 'Straight',
        '2': 'Curved',
        '3': 'Trawling',
        '4': 'Port connected',
        '5': 'Near port',
        '6': 'Anchored',
    }


    df = pd.read_csv("datasets/final_df_addedfeat.csv")
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    df['class N'] = df['class N'].astype(str)

    return df[features]

def create_and_train_model(df):

    label = 'class N'
    features = df.columns[:-1]

    X_feat = df[features]
    y = df[label].values.astype('str')

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
    return model, X_feat, y


def new_lore(data, bb):
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    data = data.loc[:, features]
    #instance = data.values[5, : -1]
    #prediction = model.predict([instance])
    instance = data.iloc[50, :-1].values
    ds = TabularDataset(data=data, class_name='class N', categorial_columns=['class N'])
    bbox = sklearn_classifier_bbox.sklearnBBox(bb)
    print(bb.predict([instance]))

    print("instance is:", instance)
    x = instance
    classifiers_generator = GenerateDecisionTrees()
    classifiers =classifiers_generator.decision_trees(X_feat, y)




    #print('model prediction is', model.predict([x]))
    #lore = TabularRandomGeneratorLore(bbox, x)
    encoder = ColumnTransformerEnc(ds.descriptor)
    surrogate = DecisionTreeSurrogate()
    generator = NewGen(bbox, ds, encoder, classifiers)


    proba_lore = Lore(bbox, ds, encoder, generator, surrogate)
    print('Prediction:', bb.predict([x]))

    rule = proba_lore.explain(x)
    print('Rule:', rule['rule'])
    print(rule)
    print('----- ')
    print('----- counterfactuals ')
    for cr in rule['counterfactuals']:
        print(cr)
        print('-----')



if __name__ == '__main__':
    res = load_data_from_csv()
    model, X_feat, y = create_and_train_model(res)
    print(model)
    new_lore(res, model)