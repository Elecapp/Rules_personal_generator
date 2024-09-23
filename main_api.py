import json
from doctest import debug

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import joblib

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

app = FastAPI(title="Frontend")

api = FastAPI(title="API", debug=True)

app.mount("/api", api)
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:8080"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@api.get("/predict")
def predict(week5_covid:str='c3', week4_covid:str='c3', week3_covid:str='c3',
            week5_mobility:str='m2', week4_mobility:str='m1', week3_mobility:str='m2',
            week2_mobility:str='m1',days_passed: float=322, duration:float=45):
    """
    Predict the number of cases for the next week
    :param week5_covid: the number of cases for the week 5

    :return: the number of cases for the next week
    """
    instance = [week5_covid, week4_covid, week3_covid, week5_mobility,
                week4_mobility, week3_mobility, week2_mobility, days_passed, duration]
    prediction = model.predict([instance])
    return {
        'instance': {
            'week5_covid': week5_covid,
            'week4_covid': week4_covid,
            'week3_covid': week3_covid,
            'week5_mobility': week5_mobility,
            'week4_mobility': week4_mobility,
            'week3_mobility': week3_mobility,
            'week2_mobility': week2_mobility,
            'days_passed': days_passed,
            'duration': duration
        },
        'prediction': prediction[0]
    }


def get_symbol(op):
    sym = re.sub(r'.*\w\s?(\S+)\s?\w.*','\\1',getattr(operator,op).__doc__)
    if re.match('^\\W+$',sym):return sym

def rule_to_dict(rule):
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

@api.get("/explain")
def explain(week5_covid:str='c3', week4_covid:str='c3', week3_covid:str='c3',
            week5_mobility:str='m2', week4_mobility:str='m1', week3_mobility:str='m2',
            week2_mobility:str='m1',days_passed: float=322, duration:float=45):
    prediction = predict(week5_covid, week4_covid, week3_covid, week5_mobility,
                week4_mobility, week3_mobility, week2_mobility, days_passed, duration)
    print('prediction', prediction)
    instance = [week5_covid, week4_covid, week3_covid, week5_mobility,
                week4_mobility, week3_mobility, week2_mobility, days_passed, duration]
    explanation = proba_lore.explain(instance)
    # convert explanation to json string using json.dumps
    exp_res = explanation.__str__()

    print('explanation', exp_res)
    print('explanation', explanation)
    print('explanation', explanation['rule'])
    print('counter rules', explanation['counterfactuals'])

    rule = rule_to_dict(explanation['rule'])
    crRules = [rule_to_dict(cr) for cr in explanation['counterfactuals']]

    return {
        'instance': {
            'week5_covid': week5_covid,
            'week4_covid': week4_covid,
            'week3_covid': week3_covid,
            'week5_mobility': week5_mobility,
            'week4_mobility': week4_mobility,
            'week3_mobility': week3_mobility,
            'week2_mobility': week2_mobility,
            'days_passed': days_passed,
            'duration': duration
        },
        'explanation': {
            'rule': rule,
            'counterfactuals': crRules
        }
    }
