import json
from doctest import debug
from typing import Literal

import fastapi
from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.surrogate import DecisionTreeSurrogate
from main import load_data_from_csv, create_and_train_model, ProbabilitiesWeightBasedGenerator

import re, operator

res = load_data_from_csv()
model_pkl_file = 'models/model.pkl'
if os.path.exists(model_pkl_file):
    model = joblib.load(model_pkl_file)
else:
    model = create_and_train_model(res)
    joblib.dump(model, model_pkl_file)

data = TabularDataset(data=res, class_name='Class_label')
bbox = sklearn_classifier_bbox.sklearnBBox(model)

encoder = ColumnTransformerEnc(data. descriptor)
surrogate = DecisionTreeSurrogate()
generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
proba_lore = Lore(bbox, data, encoder, generator, surrogate)

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

@covid_router.post("/explain")
def explain(event: CovidEvent):
    prediction = classify(event)
    #print('prediction', prediction)
    instance = event.to_list()
    explanation = proba_lore.explain(instance)
    # convert explanation to json string using json.dumps
    exp_res = explanation.__str__()

    rule = covid_rule_to_dict(explanation['rule'])
    crRules = [covid_rule_to_dict(cr) for cr in explanation['counterfactuals']]

    return {
        'instance': instance,
        'explanation': {
            'rule': rule,
            'counterfactuals': crRules
        }
    }
