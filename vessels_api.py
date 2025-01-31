import io

import fastapi
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import logging

import umap
import umap.umap_ as umap
from starlette.responses import StreamingResponse

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.neighgen import RandomGenerator, GeneticGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from main_api import rule_to_dict
from main_vessels import create_and_train_model, load_data_from_csv, generate_neighborhood, GenerateDecisionTrees, \
    VesselsGenerator, GeneticVesselsGenerator

from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.getLogger('numba').setLevel(logging.ERROR)

class VesselEvent(BaseModel):
    SpeedMinimum: float
    SpeedQ1: float
    SpeedMedian: float
    SpeedQ3: float
    DistanceStartShapeCurvature: float
    DistanceStartTrendAngle: float
    DistStartTrendDevAmplitude: float
    MaxDistPort: float
    MinDistPort: float

    def to_list(self):
        return [self.SpeedMinimum, self.SpeedQ1, self.SpeedMedian, self.SpeedQ3,
                self.DistanceStartShapeCurvature, self.DistanceStartTrendAngle, self.DistStartTrendDevAmplitude,
                self.MaxDistPort, self.MinDistPort]

class NeighborhoodRequest(BaseModel):
    vessel_event: VesselEvent
    num_samples: int
    neighborhood_types: int


df_vessels =  load_data_from_csv()

model, data_train, target_train = create_and_train_model(df_vessels)
instance = df_vessels.iloc[9, : -1].values
logger.info(f"Instance: {instance}")

predicted_class = model.predict([instance])
logger.info(f"Predicted class: {predicted_class}")
reducer = umap.UMAP(
        n_neighbors=5,
        min_dist= 0.3,
        n_components=2,
        metric='chebyshev',
        verbose=False
    )

reducer.fit(data_train)



app = FastAPI(title="Vessels API", description="API for vessels data", version="0.1")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )


@app.get("/")
async def root():
    return {"message": "Welcome to the Vessels API"}


@app.post("/classify_vessel")
async def classify_vessel(vessel_event: VesselEvent):
    logger.info(f"Received event: {vessel_event}")

    instance = vessel_event.to_list()
    predicted_class = model.predict([instance])

    return {
        "predicted_class": predicted_class[0],
        "instance": instance
    }

NEIGHB_TRAIN =          0b00001
NEIGHB_RANDOM =         0b00010
NEIGHB_CUSTOM =         0b00100
NEIGHB_GENETIC =        0b01000
NEIGHB_CUSTOM_GENETIC = 0b10000

@app.post("/neighborhood")
async def neighborhood(neigh_request: NeighborhoodRequest):
    """
        Creates a neighborhood around the given vessel event.
        The input event is a structered data containing the features of a vessel
        plus some additional parameters useful for the generation, like the number
        of samples and the neighborhood types.
        Here is an example of an input event:
        ```
        {
            "vessel_event": {
                "SpeedMinimum": 0.0,
                "SpeedQ1": 0.0,
                "SpeedMedian": 0.0,
                "SpeedQ3": 0.0,
                "DistanceStartShapeCurvature": 0.0,
                "DistanceStartTrendAngle": 0.0,
                "DistStartTrendDevAmplitude": 0.0,
                "MaxDistPort": 0.0,
                "MinDistPort": 0.0
            },
            "num_samples": 10,
            "neighborhood_types": 1
        }
        ```
        The `neighborhood_types` is an integer that represents the types of neighborhoods to be generated.
        It uses a bitmap to encode multiple types of neighborhoods. The following table shows the encoding:
        ```
        NEIGHB_TRAIN =          0b00001
        NEIGHB_RANDOM =         0b00010
        NEIGHB_CUSTOM =         0b00100
        NEIGHB_GENETIC =        0b01000
        NEIGHB_CUSTOM_GENETIC = 0b10000
        ```
        Thus, to generate the neighborhood using the training data, the `neighborhood_types` should be `1`.
        For a generation that will contain the training data, the random neighborhood, and the custom
        neighborhood, the `neighborhood_types` should be `7`.

        Below a few examples of combination that may be used:

        - `1` - only the training data
        - `2` - only random generator
        - `3` - training and random generator
        - `7` - training, random and custom generator
        - `31` - all generators

    """
    logger.info(f"Received event: {neigh_request.vessel_event}")
    logger.info(f"Number of samples: {neigh_request.num_samples}")
    # transform the integer neighborhood_types into a list of strings according to the bits
    neighborhood_type_labels = ['train', 'random', 'custom', 'genetic', 'custom_genetic']
    neighborhood_types_str = [neighborhood_type_labels[i] for i in range(5) if neigh_request.neighborhood_types & (1 << i)]

    logger.info(f"Neighborhood types: {neighborhood_types_str}")

    instance = np.array(neigh_request.vessel_event.to_list())
    # make a deep copy of the instance
    input_instance = np.copy(instance)

    predicted_class = model.predict([instance])
    logger.info(f"Predicted class: {predicted_class}")
    neighbs = generate_neighborhood(instance, model, df_vessels, data_train, target_train, neigh_request.num_samples, neighborhood_types_str)

    # create an empty data frame to aggregate the neighborhoods
    df_neighbs = pd.DataFrame([instance], columns=df_vessels.columns[:-1])
    df_neighbs['predicted_class'] = predicted_class
    df_neighbs['neighborhood_type'] = 'instance'

    for i, neighb in enumerate(neighbs):
        logger.info(f"Processing neighborhood type: {neighborhood_types_str[i]}")
        neighb_classes = model.predict(neighb)
        # create a dataframe containing all the columns of neighb, plus and the predicted class and the string label
        neighb_df = pd.DataFrame(neighb, columns=df_vessels.columns[:-1])
        if neighborhood_types_str[i] == 'train':
            neighb_df['predicted_class'] = target_train
        else:
            neighb_df['predicted_class'] = neighb_classes
        neighb_df['neighborhood_type'] = neighborhood_types_str[i]

        df_neighbs = pd.concat([df_neighbs, neighb_df], ignore_index=True)

    # apply UMAP to the neighborhood data. Only to the features contained in the original df_vessels
    embedding = reducer.transform(df_neighbs[df_vessels.columns[:-1]])
    df_neighbs['umap1'] = embedding[:, 0]
    df_neighbs['umap2'] = embedding[:, 1]



    # return df_neighbs as a csv data
    stream = io.StringIO()
    df_neighbs.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=neighborhood.csv"

    return response

@app.post("/explain")
async def explain(request:NeighborhoodRequest):
    logger.info(f"Received event: {request.vessel_event}")
    logger.info(f"Number of size of neighborhood: {request.num_samples}")
    neighborhood_type_labels = ['train', 'random', 'custom', 'genetic', 'custom_genetic']
    neighborhood_types_str = [neighborhood_type_labels[i] for i in range(5) if request.neighborhood_types & (1 << i)]
    selected_neighbor_generator = neighborhood_types_str[0]
    instance = np.array(request.vessel_event.to_list())

    predicted_class = model.predict([instance])
    logger.info(f"Predicted class: {predicted_class}")

    ds = TabularDataset(data=df_vessels, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    surrogate = DecisionTreeSurrogate()
    generator = None
    if selected_neighbor_generator == 'random':
        generator = RandomGenerator(bbox, ds, encoder)
    if selected_neighbor_generator == 'genetic':
        generator = GeneticGenerator(bbox, ds, encoder)
    if selected_neighbor_generator == 'custom':
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(data_train, target_train)
        generator = VesselsGenerator(bbox, ds, encoder, data_train, classifiers, 0.05)
    if selected_neighbor_generator == 'genetic':
        generator = GeneticGenerator(bbox, ds, encoder)
    if selected_neighbor_generator == 'custom_genetic':
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(data_train, target_train)
        generator = GeneticVesselsGenerator(bbox, ds, encoder, data_train, classifiers, 0.05)

    proba_lore = Lore(bbox, ds, encoder, generator, surrogate)

    explanation = proba_lore.explain(instance, num_instances=request.num_samples)
    # convert explanation to json string using json.dumps
    rule = rule_to_dict(explanation['rule'])
    crRules = [rule_to_dict(cr) for cr in explanation['counterfactuals']]

    return {
        'instance': instance.tolist(),
        'predicted_class': predicted_class[0],
        'explanation': {
            'rule': rule,
            'counterfactuals': crRules
        }
    }
