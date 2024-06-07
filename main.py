import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

import random
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.lore import TabularRandomGeneratorLore, Lore

from sklearn.metrics import pairwise_distances

import altair as alt

alt.data_transformers.enable('default', max_rows=None)


class IdentityEncoder(EncDec):
    """
    It provides an interface to access Identity encoding functions.
    """

    def __init__(self, dataset_descriptor):
        super().__init__(dataset_descriptor)

    def encode(self, x: np.array) -> np.array:
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        return x

    def get_encoded_features(self):
        """
        Provides a dictionary with the new encoded features name and the new index
        :return:
        """
        return self.encoded_features

    def decode(self, x: np.array) -> np.array:
        return x

    def decode_target_class(self, x: np.array):
        return x


class ProbabilitiesWeightBasedGenerator(NeighborhoodGenerator):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr)
        self.neighborhood = None
        self.preprocess = bbox.bbox.named_steps.get('columntransformer')

    def generate(self, z: np.array, num_instances: int, descriptor: dict, encoder):
        x = encoder.decode(z.reshape(1, -1))[0]
        z1 = self.preprocess.transform(x.reshape(1, -1))[0]

        perturbed_arrays = []

        choices = [0, 1, 2, 3, 4]

        covid_weights = [
            [0.0, 0.25, 0.13, 0.36, 0.26],  # 0
            [0.0, 0.40, 0.30, 0.20, 0.10],  # 1
            [0.0, 0.25, 0.40, 0.25, 0.10],  # 2
            [0.0, 0.10, 0.25, 0.40, 0.25],  # 3
            [0.0, 0.10, 0.20, 0.30, 0.40],  # 4
        ]

        mobility_weights = [
            [0.0, 0.12, 0.18, 0.45, 0.24],  # 0
            [0.0, 0.40, 0.30, 0.20, 0.10],  # 1
            [0.0, 0.25, 0.40, 0.25, 0.10],  # 2
            [0.0, 0.10, 0.25, 0.40, 0.25],  # 3
            [0.0, 0.10, 0.20, 0.30, 0.40]  # 4

        ]

        for _ in range(num_instances):
            perturbed_arr = z1.copy()

            for val in range(0, 3):  # covid
                perturbed_arr[val] = random.choices(choices, weights=covid_weights[val])[0]

            for val in range(3, 7):  # mobility
                perturbed_arr[val] = random.choices(choices, weights=mobility_weights[val - 3])[0]

            perturbed_arr[7] = random.choice(range(42, 442, 7))
            perturbed_arr[8] = random.choice(range(7, 148, 7))

            covid_sec = [f"c{int(v)}" for v in perturbed_arr[0:3]]
            mob_sec = [f"m{int(v)}" for v in perturbed_arr[3:7]]
            tot = [*covid_sec, *mob_sec, perturbed_arr[7], perturbed_arr[8]]

            dec = encoder.encode([tot])[0]

            perturbed_arrays.append(dec)

        # save to png
        self.neighborhood = np.array(perturbed_arrays)

        return self.neighborhood

    def check_generated(self, filter_function=None, check_fuction=None):
        pass


def load_data_from_csv():
    df = pd.read_csv("datasets/Final_data.csv")

    df = df.loc[:, ['Week5_Covid', 'Week4_Covid', 'Week3_Covid', 'Week5_Mobility', 'Week4_Mobility',
                    'Week3_Mobility', 'Week2_Mobility', 'Days_passed', 'Duration', 'Class_label']]

    mask = df.drop(["Class_label"], axis=1).isna().all(axis=1) & df['Class_label'].notna()
    result_df = df[~mask]
    # df = df.dropna(how = 'all')
    result_df = result_df.fillna("NONE")

    return result_df


def create_and_train_model(result_df):
    y = result_df["Class_label"].values
    X_feat = result_df.loc[:, 'Week5_Covid':'Duration'].values
    covid_categories = ['NONE', 'c1', 'c2', 'c3', 'c4']
    mobility_categories = ['NONE', 'm1', 'm2', 'm3', 'm4']
    enc = OrdinalEncoder(
        categories=[covid_categories, covid_categories, covid_categories, mobility_categories, mobility_categories,
                    mobility_categories, mobility_categories])
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("ordinal-encoder", enc, list(range(0, 7))),
            ("standard_scaler", numerical_preprocessor, list(range(7, 9))),
        ]
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    model = make_pipeline(preprocessor, clf)

    data_train, data_test, target_train, target_test = train_test_split(
        X_feat, y, random_state=0
    )
    _ = model.fit(data_train, target_train)

    encoded_train = preprocessor.fit_transform(data_train)
    df = pd.DataFrame(encoded_train)
    df.columns = result_df.columns[:-1]
    df.iloc[:, -2:] = result_df.iloc[:, -3:-1]
    df['Class_label'] = result_df['Class_label']
    print(df)

    print(model.score(data_test, target_test))

    return model


def compute_statistics_distance():
    res = load_data_from_csv()
    model = create_and_train_model(res)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    data = TabularDataset(data=res, class_name='Class_label')
    x = data.df.iloc[355, :-1].values
    encoder = ColumnTransformerEnc(data.descriptor)
    generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
    rnd_generator = RandomGenerator(bbox, data, encoder)

    print('Computing distances of custom generator')
    np_mins = measure_distances(data, encoder, generator, x)
    print('custom generator', np_mins)

    print('Computing distances of random generator')
    rnd_np_mins = measure_distances(data, encoder, rnd_generator, x)
    print('random generator', rnd_np_mins)
    alt_df_dist_o = pd.DataFrame(np_mins, columns=['Distance'])
    alt_df_dist_r = pd.DataFrame(rnd_np_mins, columns=['Distance'])
    print(alt_df_dist_o.head(10))
    alt_df_dist_o['source'] = 'our'
    alt_df_dist_r['source'] = 'random'
    boxplot_df = pd.concat([alt_df_dist_o, alt_df_dist_r], axis=0)
    box_plot = alt.Chart(boxplot_df).mark_boxplot().encode(
        alt.X("Distance:Q").scale(zero=False),
        alt.Y("source:N"),
        alt.Color("source:N")
    ).properties(
        height=300,
        width=500,
        title='Distances with the original Data'
    )
    box_plot.save('plot/boxplot.png')


def measure_distances(data, encoder, generator, x):
    global_mins = []
    for i in range(10):
        neighb = generator.generate(encoder.encode(x.reshape(1, -1))[0], 1000, data.descriptor, encoder)
        dists = calculate_distance(neighb, encoder.encoder.transform(data.df.iloc[:, :-1].values))
        local_mins = np.min(dists, axis=1)
        global_mins.append(local_mins)
        if i % 100 == 0:
            print(f"{i}...\r")
    print()
    np_mins = np.average(np.array(global_mins), axis=0)
    return np_mins


def new_lore():
    res = load_data_from_csv()

    model = create_and_train_model(res)
    instance = res.values[5, : -1]
    print(instance)
    prediction = model.predict([instance])

    print(prediction)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    data = TabularDataset(data=res, class_name='Class_label')
    x = data.df.iloc[355, :-1].values
    print(x)
    # lore = TabularRandomGeneratorLore(bbox, data)
    encoder = ColumnTransformerEnc(data.descriptor)
    surrogate = DecisionTreeSurrogate()
    generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
    proba_lore = Lore(bbox, data, encoder, generator, surrogate)

    print(data.descriptor)
    rule = proba_lore.explain(x)
    print(rule)
    print('-----')
    print(rule['rule'])
    print('-----')
    for cr in rule['counterfactuals']:
        print(cr)
        print('-----')

    # for i in range(nrows):
    #    rl = proba_lore.explain(data.df.iloc[i, :-1].values)
    #    if len(rl['counterfactuals']) > 0:
    #        print(i)


def calculate_distance(X1: np.array, X2: np.array, y_1: np.array = None, y_2: np.array = None):
    dists = pairwise_distances(X1, X2, metric='euclidean')

    return dists


def project_and_plot_data(X: np.array, y: np.array = None):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    new_lore()
    compute_statistics_distance()
