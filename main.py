import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, LabelEncoder
# from sklearn import metrics


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.compose import make_column_selector as selector
import random
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.lore import TabularRandomGeneratorLore, Lore


class IdentityEncoder(EncDec):
    """
    It provides an interface to access Identity encoding functions.
    """
    def __init__(self,dataset_descriptor):
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

    def __init__(self, bbox: AbstractBBox, dataset:Dataset, encoder:EncDec, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr)

    def generate(self, x: np.array, num_instances: int, descriptor: dict, encoder):

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
            perturbed_arr = x.copy()

            for val in range(0, 3):  # covid
                perturbed_arr[val] = random.choices(choices, weights=covid_weights[val])[0]

            for val in range(3, 7):  # mobility
                perturbed_arr[val] = random.choices(choices, weights=mobility_weights[val - 3])[0]

            perturbed_arr[7] = random.choice(range(42, 442, 7))
            perturbed_arr[8] = random.choice(range(7, 148, 7))

            perturbed_arrays.append(perturbed_arr)

        # save to png
        return np.array(perturbed_arrays)

    def check_generated(self, filter_function=None, check_fuction=None):
        pass








def load_data_from_csv():
    df = pd.read_csv("datasets/Final_data.csv")

    df = df.loc[:, ['Week5_Covid', 'Week4_Covid', 'Week3_Covid', 'Week5_Mobility', 'Week4_Mobility',
                    'Week3_Mobility', 'Week2_Mobility', 'Days_passed', 'Duration','Class_label']]

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

    print(model.score(data_test, target_test))


    return model


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
    lore = TabularRandomGeneratorLore(bbox,data)
    encoder = ColumnTransformerEnc(data.descriptor)
    surrogate = DecisionTreeSurrogate()
    generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
    proba_lore = Lore(bbox, data, encoder, generator,surrogate)

    print(data.descriptor)
    rule = proba_lore.explain(x)
    print(rule)
    print('-----')
    print(rule['rule'])
    print('-----')
    for cr in rule['counterfactuals']:
        print(cr)
        print('-----')


    #for i in range(nrows):
    #    rl = proba_lore.explain(data.df.iloc[i, :-1].values)
    #    if len(rl['counterfactuals']) > 0:
    #        print(i)





'''
def generate_neigh_from_instance():
    res = load_data_from_csv()
    model = create_and_train_model(res)
    instance = res.values[5, 1:]
    #print(instance)
    prediction = model.predict([instance])
    #print(prediction)

    cTransformer = model.named_steps.get('columntransformer')
    clf = model.named_steps.get('randomforestclassifier')
    gen = NewGen()
    neighb = gen.perturb(cTransformer.transform([instance])[0],100)
    arrays = []
    for name, indices in cTransformer.output_indices_.items():
        transformer = cTransformer.named_transformers_.get(name)
        arr = neighb[:, indices.start: indices.stop]

        #if transformer in (None, 'passthrough', 'drop'):
        #    pass
        #else:
        #    arr = transformer.inverse_transform(arr)

        arrays.append(arr)

    retarr = np.concatenate(arrays, axis=1)

    # bbox = sklearn_classifier_bbox.sklearnBBox(clf)
    #Add the predicted class to the neighbourhood array
    neigh_class_pre = clf.predict(retarr)
    neighbourhood_class = np.hstack((retarr, neigh_class_pre.reshape(-1, 1)))
    covid_categories = [['NONE', 'c1', 'c2', 'c3', 'c4']]

    enc = OrdinalEncoder(categories=covid_categories)
    last_column = neighbourhood_class[:, -1].reshape(-1, 1)
    encoded_last_column = enc.fit_transform(last_column)
    neighbourhood_class[:, -1] = encoded_last_column.flatten()

    #create the surrogate
    neighbourhood_class = neighbourhood_class.astype(int)
    surrogate = DecisionTreeSurrogate()
    surrogate.train(neighbourhood_class[:,:-1], neighbourhood_class[:,-1])

    #generate rule
    x_n = neighbourhood_class[5,:-1]
    #rule = surrogate.get_rule(x_n, neighbourhood_class)
    #print(rule)
'''



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    new_lore()







