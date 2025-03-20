import time

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

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


import random
import math

import vessels_utils
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.util import neuclidean
from lore_sa.lore import TabularRandomGeneratorLore, Lore

from sklearn.metrics import pairwise_distances, euclidean_distances

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
        attributes = vessels_utils.vessels_features

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
                         alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=30, mutpb=0.2, cxpb=0.5,
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

    def mate(self, ind1, ind2):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        This implementation uses the original implementation of the DEAP library. It adds a special case for the
        one-hot encoding, where the crossover is done taking into account the intervals of values imposed by
        the one-hot encoding.

        This overrides also keeps the blocks of the speed attributes together, to ensure that the relative values
        are maintained.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        if self.encoder.type == 'one-hot':
            intervals = [[0, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
            cxInterval1 = random.randint(0, len(intervals) - 1)
            cxInterval2 = random.randint(0, len(intervals) - 1)
            if cxInterval1 > cxInterval2:
                # Swap the two cx intervals
                cxInterval1, cxInterval2 = cxInterval2, cxInterval1

            cxpoint1 = intervals[cxInterval1][0]
            cxpoint2 = intervals[cxInterval2][1]
            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        else:
            size = min(len(ind1), len(ind2))
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        return ind1, ind2





class BaselineTrainingGenerator(NeighborhoodGenerator):
    """
    This generator serves as a baseline for the generation of rules. Since many of our rules are dependent on the
    synthetic data used for training during the generation, this generator will be used to generate the training data
    in a consistent way. This will allow us to compare the performance of the rules generated by the other generator we
    may want to implement.
    """
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1, df_train: pd.DataFrame = None):
        super().__init__(bbox, dataset, encoder, ocr)
        self.df_train = df_train

    def generate(self, x, num_instances:int=10000, descriptor: dict=None, encoder=None, list=None):
        distances = pairwise_distances(self.df_train, x.reshape(1, -1), metric='euclidean')
        # distances = self.df_train.apply(lambda row: euclidean_distances(x_, row), axis=1)
        df_neighb = self.df_train.copy()
        df_neighb['Distance'] = distances
        df_neighb.sort_values('Distance', inplace=True)

        max_rows = min(num_instances, len(df_neighb))

        return self.encoder.encode(df_neighb.iloc[:max_rows, :-1].values)




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
    features = vessels_utils.vessels_features  + ['class N']
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

def neighborhood_type_to_generators(neighborhood_types:[str], bbox, ds, encoder, data_train, target_train):
    generators = []
    if 'random' in neighborhood_types:
        random_n_generator = RandomGenerator(bbox, ds, encoder)
        generators.append(('random', random_n_generator))
    if 'custom' in neighborhood_types:
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(data_train, target_train)
        custom_generator = VesselsGenerator(bbox, ds, encoder, data_train, classifiers, 0.05)
        generators.append(('custom', custom_generator))
    if 'genetic' in neighborhood_types:
        genetic_n_generator = GeneticGenerator(bbox, ds, encoder)
        generators.append(('genetic', genetic_n_generator))
    if 'custom_genetic' in neighborhood_types:
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(data_train, target_train)
        custom_gen_generator = GeneticVesselsGenerator(bbox, ds, encoder, data_train, classifiers, 0.05)
        generators.append(('custom_genetic', custom_gen_generator))
    if 'baseline' in neighborhood_types:
        baseline_generator = BaselineTrainingGenerator(bbox, ds, encoder, df_train=data_train)
        generators.append(('baseline', baseline_generator))

    return generators

def generate_neighborhoods(x, model, data, X_feat, y, num_instances=100, neighborhood_types=['train', 'random']):
    """
    Generate neighborhoods for a given instance x and model. The generators are selected from the neighborhood_types.
    Each label corresponds to one of the generators avaialble for this case study. See the function `neighborhood_type_to_generators`
    for more details.

    :param x: The instance for which the neighborhoods are generated
    :param model: The model used for the prediction
    :param data: The data used for the statistics of the training data. It is used to set up the generators metadata.
    :param X_feat: The features of the data to be used as training set. This is used when the 'train' neighborhood is selected.
    :param y: The target labels of the data to be used as training set. This is used when the 'train' neighborhood is selected.
    :param num_instances: The number of instances to generate for each neighborhood generation
    :param neighborhood_types: The types of neighborhoods to generate. The available types are: 'train', 'random', 'custom', 'genetic', 'custom_genetic', 'baseline'

    :return: A list of tuples with the neighborhood type and the generated neighborhoods. The tuple is in the form (neighborhood_type, neighborhoods)
    """
    # Check if the neighborhoods already exist
    ds = TabularDataset(data=data, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    result = ()

    if 'train' in neighborhood_types:
        result = result + (('train', X_feat.values),)

    generators = neighborhood_type_to_generators(neighborhood_types, bbox, ds, encoder, X_feat, y)
    for (n, g) in generators:
        gen_neighb = g.generate(x, num_instances, ds.descriptor, encoder)
        result = result + ((n, gen_neighb), )
    return result

def generate_neighborhood_statistics(x, model, data, X_feat, y, num_instances=100,
                                     num_repeation=10, neighborhood_types=['train', 'random'], an_array=None):

    ds = TabularDataset(data=data, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    result = ()
    compl_times = ()

    generators = neighborhood_type_to_generators(neighborhood_types, bbox, ds, encoder, X_feat, y)
    for (n, g) in generators:
        print(f'Generator: {n}')
        global_mins_training = []
        global_mins_instance = []
        global_times = []
        for i in range(num_repeation):
            if i % 2 == 0:
                print(f"Repetition {i}")
            start = time.time()
            gen_neighb = g.generate(x.copy(), num_instances, ds.descriptor, encoder)
            gen_neighb = gen_neighb[:num_instances]
            end = time.time()
            dists_training = compute_distance(gen_neighb, an_array)
            dists_instance = compute_distance(gen_neighb, x.reshape(1, -1))
            global_mins_training.append(dists_training)
            global_mins_instance.append(dists_instance)
            global_times.append(end - start)

        np_mean_training = np.mean(np.array(global_mins_training), axis=0)
        np_mean_instance = np.mean(np.array(global_mins_instance), axis=0)
        result = result + ((n, 'training', np_mean_training), )
        result = result + ((n, 'instance', np_mean_instance), )
        compl_times = compl_times + ((n, global_times),)
    return result, compl_times



def compute_distance(X1, X2, metric:str='euclidean'):
    dists = pairwise_distances(X1, X2, metric=metric)
    dists = np.min(dists, axis=1)
    return dists

def measure_distance(neighborhoods, an_array):# an_array can be a point or X_feat
    """
    Measure the distance of 'an_array' to the neighborhoods. The neighborhoods are in the form of a list of tuples
    where the first element is the neighborhood type and the second element is the neighborhood data.

    :param neighborhoods: The neighborhoods to measure the distance to
    :param an_array: The array to measure the distance to the neighborhoods

    :return: A list of tuples with the neighborhood type and the distances to the neighborhoods. The tuple is in the form (neighborhood_type, distances)
    """
    distances = []
    for n, neigh in neighborhoods:
        dists = compute_distance(neigh, an_array)
        distances.append((n, dists))

    return distances



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

def plot_boxplot(df, df_times, basefilename):
    #domain_ = ['Random', 'Custom', 'Genetic', 'GPT']
    #range_ = ['#102ce0', '#fa7907', '#027037', '#FF5733']

    text_times = alt.Chart(df_times).mark_text(
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        text='mean(Time):Q',
        y=alt.Y('Neighborhood:N'),
    ).configure_view(
        stroke=None
    )
    box_plot_times = text_times

    box_plot_times.save(f'plot/instance_vs_neigh/{basefilename}_times.pdf')

    box_plot_training = alt.Chart(df[df['Reference']=='training']).mark_boxplot().encode(
        x=alt.X("Distance:Q"),
        y=alt.Y("Neighborhood:N"),
        color=alt.Color("Neighborhood:N") #scale=alt.Scale(domain=domain_, range=range_)
    ).properties(
        height=200,
        width=200,
        title='Euclidean distance to training data'
    )
    box_plot_training.save(f'plot/instance_vs_neigh/{basefilename}_training.pdf')

    box_plot_instance = alt.Chart(df[df['Reference']=='instance']).mark_boxplot().encode(
        x=alt.X("Distance:Q"),
        y=alt.Y("Neighborhood:N"),
        color=alt.Color("Neighborhood:N") #scale=alt.Scale(domain=domain_, range=range_)
    ).properties(
        height=200,
        width=200,
        title='Euclidean distance to instance'
    )
    box_plot_instance.save(f'plot/instance_vs_neigh/{basefilename}_instance.pdf')

    box_plot_both = box_plot_instance | box_plot_training
    box_plot_both.save(f'plot/instance_vs_neigh/{basefilename}_both.pdf')


if __name__ == '__main__':
    res = load_data_from_csv()
    model, X_feat, y = create_and_train_model(res)
    print(model)

    # id_instance = 192
    # instance = res.iloc[id_instance, :-1].values  # Example instance

    id_instance = 1003
    instance = np.array([0.91, 2.37, 2.49, 2.78, 2.45, -0.01, 5.07, 27.55, 20.04])  # Example instance

    # result_n = generate_neighborhoods(instance, model, res, X_feat, y, num_instances=100, neighborhood_types=['train', 'random',])
    print(f'Instance {id_instance} is: {instance}')

    #new_lore(res, model)
    #instance = res.iloc[9, :-1].values
    #distances_instance = measure_distance(result_n, instance.reshape(1, -1))
    #print(random_distance, custom_distance, genetic_distance)

    #distances_train = measure_distance(result_n, X_feat.values)
    #Example: extracting a column

    num_repeation = 50
    num_instances = 2000
    neighborhood_types = [
        'train',
        'random',
        'custom',
        'genetic',
        'custom_genetic',
        'baseline'
    ]

    basefilename = f'vessels_neighborhoods_x_{id_instance}_{num_instances}_{num_repeation}rep'
    csv = f'{basefilename}.csv'
    if not os.path.exists('%s' % csv):
        dists, compl_times = generate_neighborhood_statistics(instance, model, res, res.loc[:,vessels_utils.vessels_features], res['class N'], num_instances=num_instances,
                                                 num_repeation=num_repeation,
                                                 neighborhood_types=neighborhood_types, an_array=X_feat.values)
        df = pd.DataFrame([], columns=['Neighborhood', 'Reference', 'Distance'])
        df_times = pd.DataFrame([], columns=['Neighborhood', 'Time'])
        for (n, t, d) in dists:
            df = pd.concat([df, pd.DataFrame({'Neighborhood': [n] * len(d), 'Reference': [t] * len(d),'Distance': d})], axis=0)
        for (n, t) in compl_times:
            df_times = pd.concat([df_times, pd.DataFrame({'Neighborhood': [n] * len(t), 'Time': t})], axis=0)
        df.to_csv(csv, index=False)
        df_times.to_csv(f'{basefilename}_times.csv', index=False)
    else:
        df = pd.read_csv(csv)
        df_times = pd.read_csv(f'{basefilename}_times.csv')

    plot_boxplot(df, df_times, basefilename)





