import io
import json
import logging
from doctest import debug
from typing import Literal, List

import fastapi
import numpy as np
import pandas as pd
import umap
from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from starlette.responses import StreamingResponse, Response

from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.encoder_decoder import ColumnTransformerEnc, EncDec
from lore_sa.lore import Lore
from lore_sa.neighgen import RandomGenerator, GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.util import neuclidean
from main import load_data_from_csv, create_and_train_model, ProbabilitiesWeightBasedGenerator, GPTCovidGenerator
import altair as alt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('numba').setLevel(logging.ERROR)



res = load_data_from_csv()
model_pkl_file = 'models/model.pkl'
if os.path.exists(model_pkl_file):
    model = joblib.load(model_pkl_file)
else:
    model = create_and_train_model(res)
    joblib.dump(model, model_pkl_file)

transformer = model.named_steps['columntransformer']
reducer = Pipeline(steps=[
    ('columntransformer', transformer),
    ('reducer', umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, metric='chebyshev', verbose=False))
])


data = TabularDataset(data=res, class_name='Class_label')
bbox = sklearn_classifier_bbox.sklearnBBox(model)



reducer.fit(res[res.columns[:-1]])


covid_router = fastapi.APIRouter(
    prefix="/covid",
    tags=["COVID"],
)


def dataframe_to_vega(df):
    attributes = ['Week6_Covid', 'Week5_Covid', 'Week4_Covid', 'Week3_Covid', 'Week2_Covid', 'Week6_Mobility',
          'Week5_Mobility', 'Week4_Mobility', 'Week3_Mobility', 'Week2_Mobility', 'Week1_Mobility', 'Days_passed']
    # create a nominal colro scale for the neighborhood types
    color_scale = alt.Scale(domain=['instance', 'train', 'random', 'custom', 'genetic', 'gpt', 'baseline'],
                            range=['#333333', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#e5c494'])
    # create a chart of the projected points
    #brush = alt.selection_interval(
    #     on="[pointerdown[event.altKey], pointerup] > pointermove",
    #     name='brush'
    # )
    bind = alt.selection_interval(bind='scales')
    chartUMAP = alt.Chart(df).mark_point().encode(
        x='umap1:Q',
        y='umap2:Q',
        color=alt.when(bind).then(alt.Color('neighborhood_type:N', scale=color_scale)).otherwise(
            alt.value('lightgray')),
        shape='predicted_class:N',
        tooltip=attributes + ['predicted_class', 'neighborhood_type']
    ).properties(
        width=600,
        height=600,
        title='UMAP projection of the Covid data'
    )
    chartUMAP = (chartUMAP.transform_filter(alt.datum.neighborhood_type != 'instance').add_params(bind)
                 + chartUMAP.transform_filter(alt.datum.neighborhood_type == 'instance')
                 )
    chartClasses = (alt.Chart(df).mark_bar().encode(
        x='predicted_class:N',
        y='count()',
        color=alt.Color('neighborhood_type:N', scale=color_scale),
        column='neighborhood_type:N',
        tooltip=['predicted_class', 'count()']
    ).transform_filter(alt.datum.neighborhood_type != 'instance')
    .transform_filter(bind)  # filter the data based on the brush selection
    .properties(
        width=200,
        height=200
    ))
    marginalCharts = (alt.Chart(df).mark_bar().encode(
        y=alt.Y('count()', title=''),
        color=alt.Color('neighborhood_type:N', scale=color_scale),
        column=alt.Column('neighborhood_type:N', title=None)
    ).transform_filter(alt.datum.neighborhood_type != 'instance').properties(
        width=200,
        height=100
    ).transform_filter(bind))
    attributeCharts = []
    for attribute in attributes:
        if attribute == "Days_passed":
            attributeBarChart = marginalCharts.encode(
                x=alt.X(attribute, title=attribute)
                .bin(maxbins=20),
            ).properties(
                title=attribute,
            )
        else:
            attributeBarChart = marginalCharts.encode(
                x=alt.X(attribute, type="ordinal", title=attribute),
            ).properties(
                title=attribute,
            )
        attributeCharts.append(attributeBarChart)
    neighbsCharts = alt.vconcat(*attributeCharts)
    dashboard = (alt.vconcat(chartUMAP, chartClasses, neighbsCharts))
    return dashboard

class CovidEvent(BaseModel):
    week6_covid: Literal['c1', 'c2', 'c3', 'c4']
    week5_covid: Literal['c1', 'c2', 'c3', 'c4']
    week4_covid: Literal['c1', 'c2', 'c3', 'c4']
    week3_covid: Literal['c1', 'c2', 'c3', 'c4']
    week2_covid: Literal['c1', 'c2', 'c3', 'c4']
    week6_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week5_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week4_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week3_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week2_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week1_mobility: Literal['m1', 'm2', 'm3', 'm4']
    days_passed: float

    def to_list(self):
        return [self.week6_covid, self.week5_covid, self.week4_covid, self.week3_covid, self.week2_covid,
                self.week6_mobility, self.week5_mobility, self.week4_mobility, self.week3_mobility,
                self.week2_mobility, self.week1_mobility, self.days_passed]

class CovidRequest(BaseModel):
    event: CovidEvent
    num_samples: int = 500
    neighborhood_types: List[Literal['train', 'random', 'custom', 'genetic', 'gpt', 'baseline']]


@covid_router.post("/classify")
def classify(event: CovidEvent):
    """
    Predict the number of cases for the next week
    :param week5_covid: the number of cases for the week 5

    :return: the number of cases for the next week
    """
    instance = event.to_list()
    prediction = model.predict([instance])
    return {
        'instance': instance,
        'prediction': prediction[0]
    }

def covid_rule_to_dict(rule):
    premises = [{'attr': e.variable, 'val': e.value, 'op': e.operator2string()}
                for e in rule.premises]


    return {
        'premises': premises,
        'consequence': {
            'attr': rule.consequences.variable,
            'val': rule.consequences.value,
            'op': rule.consequences.operator2string()
        }
    }

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
        z_train =self.encoder.encode (self.df_train.values)
        distances = [neuclidean(x, row) for row in z_train]
        df_neighb = self.df_train.copy()
        df_neighb['Distance'] = distances
        df_neighb.sort_values('Distance', inplace=True)

        max_rows = min(num_instances, len(df_neighb))

        return self.encoder.encode(df_neighb.iloc[:max_rows, :-1].values)

def neighborhood_type_to_generators(neighborhood_types:[str], bbox, ds, encoder):
    generators = []
    if 'random' in neighborhood_types:
        random_n_generator = RandomGenerator(bbox, ds, encoder)
        generators.append(('random', random_n_generator))
    if 'custom' in neighborhood_types:
        custom_generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
        generators.append(('custom', custom_generator))
    if 'genetic' in neighborhood_types:
        genetic_n_generator = GeneticGenerator(bbox, ds, encoder)
        generators.append(('genetic',genetic_n_generator))
    if 'gpt' in neighborhood_types:
        gpt_generator = GPTCovidGenerator(bbox, data, encoder)
        generators.append(('gpt', gpt_generator))
    if 'baseline' in neighborhood_types:
        baseline_generator = BaselineTrainingGenerator(bbox, ds, encoder, 0.1, ds.df[ds.df.columns[:-1]])
        generators.append(('baseline', baseline_generator))

    return generators

@covid_router.post("/neighborhood")
async def neighborhood(request: CovidRequest):
    logger.info(f"Received event: {request.event}")
    logger.info(f"Number of samples: {request.num_samples}")
    logger.info(f"Neighborhood types: {request.neighborhood_types}")

    df_neighbs = await compute_neighborhoods(request)

    stream = io.StringIO()
    df_neighbs.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=neighborhood.csv"

    return response

@covid_router.post("/neighborhood/visualization")
async def neighborhood_visualization(request: CovidRequest):
    logger.info(f"Received event: {request.event}")
    logger.info(f"Number of samples: {request.num_samples}")
    logger.info(f"Neighborhood types: {request.neighborhood_types}")

    df_neighbs = await compute_neighborhoods(request)

    dashboard = dataframe_to_vega(df_neighbs)
    return Response(content=dashboard.to_json(), media_type="application/json")


async def compute_neighborhoods(request):
    instance_event = (request.event.to_list())
    predicted_class = model.predict([instance_event])
    logger.info(f"Predicted class: {predicted_class}")
    encoder = ColumnTransformerEnc(data.descriptor)
    z = encoder.encode([instance_event])[0]
    generators = neighborhood_type_to_generators(request.neighborhood_types, bbox, data, encoder)
    df_neighbs = pd.DataFrame([instance_event], columns=res.columns[:-1])
    df_neighbs['predicted_class'] = predicted_class
    df_neighbs['neighborhood_type'] = 'instance'
    df_train = res.copy()[res.columns[:-1]]
    df_train['predicted_class'] = res[res.columns[-1]]
    df_train['neighborhood_type'] = 'train'
    df_neighbs = pd.concat([df_neighbs, df_train], ignore_index=True)
    for (n, gen) in generators:
        logger.info(f"Processing neighborhood type: {n}")

        neighbs_z = gen.generate(z, request.num_samples, data.descriptor, encoder)
        neighbs = encoder.decode(neighbs_z)
        neighb_classes = model.predict(neighbs)
        neighb_df = pd.DataFrame(neighbs, columns=res.columns[:-1])
        neighb_df['predicted_class'] = neighb_classes
        neighb_df['neighborhood_type'] = n
        df_neighbs = pd.concat([df_neighbs, neighb_df], ignore_index=True)
    embedding = reducer.transform(df_neighbs[res.columns[:-1]])
    df_neighbs['umap1'] = embedding[:, 0]
    df_neighbs['umap2'] = embedding[:, 1]
    return df_neighbs


@covid_router.post("/explain")
def explain(request: CovidRequest):
    prediction = classify(request.event)
    #print('prediction', prediction)
    instance = request.event.to_list()
    neighborhood_types_str = request.neighborhood_types


    encoder = ColumnTransformerEnc(data. descriptor)
    surrogate = DecisionTreeSurrogate()

    generators = neighborhood_type_to_generators(neighborhood_types_str, bbox, data, encoder)

    explanations = {}
    for (n, gen) in generators:
        spec_lore = Lore(bbox, data, encoder, gen, surrogate)
        explanation = spec_lore.explain(instance, num_instances=request.num_samples)
        # convert explanation to json string using json.dumps
        # rule = covid_rule_to_dict(explanation['rule'])
        # crRules = [covid_rule_to_dict(cr) for cr in explanation['counterfactuals']]
        explanations[n] = explanation

    return {
        'instance': instance,
        'predicted_class': prediction,
        'explanations': explanations
    }

