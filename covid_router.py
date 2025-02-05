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
from starlette.responses import StreamingResponse

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.neighgen import RandomGenerator, GeneticGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from main import load_data_from_csv, create_and_train_model, ProbabilitiesWeightBasedGenerator

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

class CovidEvent(BaseModel):
    week5_covid: Literal['c1', 'c2', 'c3', 'c4']
    week4_covid: Literal['c1', 'c2', 'c3', 'c4']
    week3_covid: Literal['c1', 'c2', 'c3', 'c4']
    week5_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week4_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week3_mobility: Literal['m1', 'm2', 'm3', 'm4']
    week2_mobility: Literal['m1', 'm2', 'm3', 'm4']
    days_passed: float
    duration: float

    def to_list(self):
        return [self.week5_covid, self.week4_covid, self.week3_covid,
                self.week5_mobility, self.week4_mobility, self.week3_mobility, self.week2_mobility,
                self.days_passed, self.duration]


class CovidRequest(BaseModel):
    event: CovidEvent
    num_samples: int = 500
    neighborhood_types: List[Literal['train', 'random', 'custom', 'genetic', 'custom_genetic']]


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


def neighborhood_type_to_generators(neighborhood_types:[str], bbox, ds, encoder):
    generators = []
    if 'random' in neighborhood_types:
        random_n_generator = RandomGenerator(bbox, ds, encoder)
        generators.append(random_n_generator)
    if 'custom' in neighborhood_types:
        custom_generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
        generators.append(custom_generator)
    if 'genetic' in neighborhood_types:
        genetic_n_generator = GeneticGenerator(bbox, ds, encoder)
        generators.append(genetic_n_generator)

    return generators

@covid_router.post("/neighborhood")
async def neighborhood(request: CovidRequest):
    logger.info(f"Received event: {request.event}")
    logger.info(f"Number of samples: {request.num_samples}")
    logger.info(f"Neighborhood types: {request.neighborhood_types}")

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

    for i, gen in enumerate(request.neighborhood_types):
        generator = generators[i]
        logger.info(f"Processing neighborhood type: {gen}")

        neighbs_z = generator.generate(z, request.num_samples, data.descriptor, encoder)
        neighbs = encoder.decode(neighbs_z)
        neighb_classes = model.predict(neighbs)
        neighb_df = pd.DataFrame(neighbs, columns=res.columns[:-1])
        neighb_df['predicted_class'] = neighb_classes
        neighb_df['neighborhood_type'] = gen
        df_neighbs = pd.concat([df_neighbs, neighb_df], ignore_index=True)

    embedding = reducer.transform(df_neighbs[res.columns[:-1]])
    df_neighbs['umap1'] = embedding[:, 0]
    df_neighbs['umap2'] = embedding[:, 1]

    stream = io.StringIO()
    df_neighbs.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=neighborhood.csv"

    return response


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
    for gen in generators:
        spec_lore = Lore(bbox, data, encoder, gen, surrogate)
        explanation = spec_lore.explain(instance, num_instances=request.num_samples)
        # convert explanation to json string using json.dumps
        # rule = covid_rule_to_dict(explanation['rule'])
        # crRules = [covid_rule_to_dict(cr) for cr in explanation['counterfactuals']]
        explanations[gen.__class__.__name__] = explanation

    return {
        'instance': instance,
        'predicted_class': prediction,
        'explanations': explanations
    }

