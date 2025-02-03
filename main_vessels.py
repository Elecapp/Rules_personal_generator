import pandas as pd
import numpy as np

import os

from deap import creator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector


import umap
import umap.umap_ as umap


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
from lore_sa.neighgen import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.util import neuclidean
from lore_sa.lore import TabularRandomGeneratorLore, Lore

from sklearn.metrics import pairwise_distances

import altair as alt
alt.data_transformers.enable('default', max_rows=None)



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



class VesselsGenerator(NeighborhoodGenerator):
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, X_feat, classifiers: dict, prob_of_mutation: float, ocr=0.1 ):
        super().__init__(bbox, dataset, encoder, ocr)
        self.neighborhood = None
        self.preprocess = bbox.bbox.named_steps.get('columntransformer')
        self.gen_data = None
        self.X_feat = X_feat
        self.classifiers = classifiers
        self.prob_of_mutation = prob_of_mutation


    def generate(self, x, num_instances:int=10000, descriptor: dict=None, encoder=None, list=None):
        perturbed_list = []
        perturbed_list.append(x.copy())

        for _ in range(num_instances):
            perturbed_x = self.perturbate(x)
            perturbed_list.append(perturbed_x)
        self.neighborhood = self.encoder.encode(perturbed_list)
        return self.neighborhood

    def perturbate(self, instance):
        descriptor = self.dataset.descriptor
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
                    (self.X_feat.columns[feature[node]], threshold[node], instance[feature[node]], instance[feature[node]] < threshold[node], node)
                )
        # extract the feature indices
        feature_indices = [f[4] for f in influencing_features]

        # make a deep copy of x in the variable perturbed_arr
        attributes = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']

        perturbed_arr = instance.copy()
        for i, v in enumerate(perturbed_arr):
            feat_name = attributes[i]
            distribution = descriptor['numeric'][feat_name]
            iqr = distribution['q3'] - distribution['q1']
            noise = np.random.normal(0, 0.2 * iqr)
            perturbed_arr[i] = np.clip(v + noise, distribution['min'], distribution['max'])

        # impose constraint of the first features
        perturbed_arr[1] = np.clip(perturbed_arr[1], perturbed_arr[0], perturbed_arr[1])
        perturbed_arr[2] = np.clip(perturbed_arr[2], perturbed_arr[1], perturbed_arr[2])
        perturbed_arr[3] = np.clip(perturbed_arr[3], perturbed_arr[2], perturbed_arr[3])


        # perturbed_arr[0] = random.uniform(0, 17.93)
        # perturbed_arr[1] = random.uniform(perturbed_arr[0], 20.12)  # always greater than minspeed
        # perturbed_arr[2] = random.uniform(perturbed_arr[1], 20.75)  # always greater than speedQ1
        # perturbed_arr[3] = random.uniform(perturbed_arr[2], 21.65)  # always greater than speedMedian
        # perturbed_arr[4] = random.uniform(0, 2.24)
        # perturbed_arr[5] = random.uniform(-0.24, 0.36)
        # perturbed_arr[6] = random.uniform(-2.80, 1.77)
        # perturbed_arr[7] = random.uniform(0.12, 282.26)
        perturbed_arr[8] = math.log10(random.uniform(math.exp(-3.05), perturbed_arr[7]))  # inverse log transform, identify value less than max dist, then log transform again

        mask_indices = [perturbed_arr[i] if (i in feature_indices or random.random() < self.prob_of_mutation) else instance[i] for i, v in enumerate(instance)] # change importance feature and some of the feature randomly based on prob
        mask_indices_arr = np.array(mask_indices, dtype=float)

        return mask_indices_arr



class GeneticVesselsGenerator(GeneticGenerator):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, X_feat, classifiers: dict, prob_of_mutation: float, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr,
                         alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                         tournsize=3, halloffame_ratio=0.1, random_seed=None
                         )
        self.vessels_generator = VesselsGenerator(bbox, dataset, encoder, X_feat, classifiers, prob_of_mutation, ocr)

    def mutate(self, toolbox, x):
        """
         This function specializes the mutation operator of the genetic algorithm to explicitly
         use the implementation of the perturbation of the VesselsGenerator class. This guarantees
            that the perturbation is consistent with the perturbation of the VesselsGenerator class.
        """
        z = toolbox.clone(x)
        z1 = self.vessels_generator.perturbate(z)

        for i,v in enumerate(z1):
            z[i] = v

        return z,

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
          that the soft rules are maintained)
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

    print('Model score: ', model.score(data_test, target_test))
    return model, data_train, target_train



def visualize_neighborhoods(random_distance, custom_distance, genetic_distance, custom_genetic_distance, title='Neighborhood Boxplots'):

    neighborhoods=[]
    for data, label in zip([random_distance, custom_distance, genetic_distance, custom_genetic_distance],['Random', 'Custom', 'Genetic', 'Custom Genetic']):
        df = pd.DataFrame(data)
        df['Neighborhood'] = label
        neighborhoods.append(df)

        # Combine all data
    combined_df = pd.concat(neighborhoods, ignore_index=True)
    combined_df.rename(columns={0: 'Distance'}, inplace=True)




    chart = alt.Chart(combined_df).mark_boxplot().encode(
            x=alt.X('Distance:Q', title='Distance'),
            y=alt.Y('Neighborhood:N'),
            color=alt.Color('Neighborhood:N', legend=alt.Legend(title='Neighborhood'))
        ).properties(
            title=title,
            width=600,
            height=400
        )
    directory = "plot"
    chart_path = os.path.join(directory, "chart.png")
    alt.renderers.enable("browser")
    #chart.save(chart_path, dpi=250)
    return chart




def generate_neighborhood(x, model, data, X_feat, y, num_instances=100, neighborhood_types=['train', 'random']):

    # Check if the neighborhoods already exist
    ds = TabularDataset(data=data, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    result = ()


    if 'train' in neighborhood_types:
        result = result + (X_feat.values,)
    if 'random' in neighborhood_types:
        random_n_generator = RandomGenerator(bbox, ds, encoder)
        random_n = random_n_generator.generate(x, num_instances, ds.descriptor, encoder)
        result = result + (random_n,)
    if 'custom' in neighborhood_types:
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(X_feat, y)
        custom_generator = VesselsGenerator(bbox, ds, encoder, X_feat, classifiers, 0.05)
        custom_n = custom_generator.generate(x, num_instances, ds.descriptor, encoder)
        result = result + (custom_n,)
    if 'genetic' in neighborhood_types:
        genetic_n_generator = GeneticGenerator(bbox, ds, encoder)
        genetic_n = genetic_n_generator.generate(x, num_instances, ds.descriptor, encoder)
        result = result + (genetic_n,)
    if 'custom_genetic' in neighborhood_types:
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(X_feat, y)
        custom_gen_generator = GeneticVesselsGenerator(bbox, ds, encoder, X_feat, classifiers, 0.05)
        custom_gen_n = custom_gen_generator.generate(x, num_instances, ds.descriptor, encoder)
        result = result + (custom_gen_n,)


    return result



def compute_distance(X1, X2, metric:str='euclidean'):
    dists = pairwise_distances(X1, X2, metric=metric)
    dists = np.min(dists, axis=1)
    return dists

def measure_distance(random_n, custom_n, genetic_n, custom_genetic_n, an_array):# an_array can be a point or X_feat
    #an_array = an_array.reshape(1, -1)
    random_distance = compute_distance(random_n, an_array)
    custom_distance = compute_distance(custom_n, an_array)
    genetic_distance = compute_distance(genetic_n, an_array)
    custom_genetic_distance = compute_distance(custom_genetic_n, an_array)

    return random_distance, custom_distance, genetic_distance, custom_genetic_distance





def new_lore(data, bb):
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    data = data.loc[:, features]
    #instance = data.values[5, : -1]
    #prediction = model.predict([instance])
    instance = data.iloc[9, :-1].values
    ds = TabularDataset(data=data, class_name='class N', categorial_columns=['class N'])
    bbox = sklearn_classifier_bbox.sklearnBBox(bb)
    print(bb.predict([instance]))

    print("instance is:", instance)
    x = instance
    classifiers_generator = GenerateDecisionTrees()
    classifiers = classifiers_generator.decision_trees(X_feat, y)

    #print('model prediction is', model.predict([x]))
    #lore = TabularRandomGeneratorLore(bbox, x)![](../../Desktop/Schermata 2024-12-23 alle 15.10.01.png)
    encoder = ColumnTransformerEnc(ds.descriptor)
    surrogate = DecisionTreeSurrogate()

    generator = VesselsGenerator(bbox, ds, encoder, classifiers, 0.05)


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


def visualize_umap(X_train, *neigh_arrays, labels=None):

    reducer = umap.UMAP(n_neighbors=100, min_dist=0.3, metric='manhattan', random_state=42)
    # da cambiare i n_neigh per vedere il migliore
    X_train_embedded = reducer.fit_transform(X_train) #only on the trainig data

    transformed_arrays = [reducer.transform(array) for array in neigh_arrays]
    #visualize the umap
    plt.figure(figsize=(10, 8))
    plt.scatter(
        X_train_embedded[:, 0],
        X_train_embedded[:, 1],
        c='#4f4e4d',
        label=labels[0] if labels else 'X_train',
        alpha=0.2,
    )

    # Plot the other arrays with different colors
    colors = ['#fc3503', '#fcca03', '#3f0cf5']  # Extend as needed
    for i, array_embedded in enumerate(transformed_arrays):
        label = labels[i + 1] if labels else f'Array {i + 1}'
        plt.scatter(
            array_embedded[:, 0],
            array_embedded[:, 1],
            c=colors[i % len(colors)],
            label=label,
            alpha=0.5,
        )

    # Add plot details
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='best')
    plt.grid(True)

    # Show plot
    plt.show()

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(f'precomputed_csv/{filename}.csv')


if __name__ == '__main__':
    res = load_data_from_csv()
    model, X_feat, y = create_and_train_model(res)
    print(model)

    save_dir = "neighborhood_cache"
    instance = res.iloc[9, :-1].values  # Example instance
    #file_path = os.path.join(save_dir, f"neighborhood_instance_{str(instance)}.pkl")
    result_n = generate_neighborhood(instance, model, res, X_feat, y, num_instances=1000, neighborhood_types=['train', 'random', 'custom'])


    #new_lore(res, model)
    #instance = res.iloc[9, :-1].values
    random_distance_inst, custom_distance_inst, genetic_distance_inst, custom_genetic_distance_inst = measure_distance(random_n, custom_n, genetic_n, custom_genetic_n, instance.reshape(1, -1))
    #print(random_distance, custom_distance, genetic_distance)

    random_distance_train, custom_distance_train, genetic_distance_train, custom_genetic_distance_train = measure_distance(random_n, custom_n, genetic_n, custom_genetic_n, X_feat.values)
    visualize_neighborhoods(random_distance_train, custom_distance_train, genetic_distance_train, custom_genetic_distance_train, title='Distances from Trainset').show()

    save_to_csv(X_feat,"X_feat")
    save_to_csv(random_n, "random_n")
    save_to_csv(custom_n,"custom_n")
    save_to_csv(genetic_n,"genetic_n")
    save_to_csv(custom_genetic_n,"custom_genetic_n")

    #Example: extracting a column
    chart = visualize_neighborhoods(random_distance_inst, custom_distance_inst, genetic_distance_inst, custom_genetic_distance_inst, title='Distances from Instance')
    chart.show()
    #visualize_umap(X_feat, random_n, custom_n, genetic_n, labels=["Train","random","custom","genetic"])





