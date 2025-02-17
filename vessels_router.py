from typing_extensions import Unpack

import logging
from typing import List, Literal

import numpy as np
import pandas as pd
import io

import fastapi
import umap
import umap.umap_ as umap
from starlette.responses import StreamingResponse, JSONResponse, Response

import altair as alt

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.neighgen import RandomGenerator, GeneticGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from covid_router import covid_rule_to_dict
from main_vessels import create_and_train_model, load_data_from_csv, generate_neighborhoods, GenerateDecisionTrees, \
    VesselsGenerator, GeneticVesselsGenerator, neighborhood_type_to_generators

from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.getLogger('numba').setLevel(logging.ERROR)

neighborhood_type_labels = ['train', 'random', 'custom', 'genetic', 'custom_genetic']

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

class VesselRequest(BaseModel):
    vessel_event: VesselEvent
    num_samples: int = 500
    neighborhood_types: List[Literal['train', 'random', 'custom', 'genetic', 'custom_genetic', 'baseline']]


# ===================================================================================
# Load the vessels data and create the model

df_vessels =  load_data_from_csv()

vessels_model, vessels_data_train, vessels_target_train = create_and_train_model(df_vessels)
instance = df_vessels.iloc[9, : -1].values
logger.info(f"Instance: {instance}")

predicted_class = vessels_model.predict([instance])
logger.info(f"Predicted class: {predicted_class}")
reducer = umap.UMAP(
        n_neighbors=5,
        min_dist= 0.3,
        n_components=2,
        metric='chebyshev',
        verbose=False
    )

reducer.fit(vessels_data_train)

# ===================================================================================

vessels_router = fastapi.APIRouter(
    prefix="/vessels",
    tags=["Vessels"],
)


def dataframe_to_vega(df):
    attributes = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                  'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']
    # create a nominal colro scale for the neighborhood types
    color_scale = alt.Scale(domain=['instance', 'train', 'random', 'custom', 'genetic', 'custom_genetic', 'baseline'],
                            range=['#333333', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#e5c494'])
    # create a chart of the projected points
    # brush = alt.selection_interval(
    #     on="[pointerdown[event.altKey], pointerup] > pointermove",
    #     name='brush'
    # )
    brush = alt.selection_interval(bind='scales')
    chartUMAP = alt.Chart(df).mark_point().encode(
        x='umap1:Q',
        y='umap2:Q',
        color=alt.when(brush).then(alt.Color('neighborhood_type:N', scale=color_scale)).otherwise(
            alt.value('lightgray')),
        shape='predicted_class:N',
        tooltip=attributes + ['predicted_class', 'neighborhood_type']
    ).properties(
        width=600,
        height=600,
        title='UMAP projection of the Vessels data'
    )
    chartUMAP = (chartUMAP.transform_filter(alt.datum.neighborhood_type != 'instance').add_params(brush)
                 + chartUMAP.transform_filter(alt.datum.neighborhood_type == 'instance')
                 )
    chartClasses = (alt.Chart(df).mark_bar().encode(
        x='predicted_class:N',
        y='count()',
        color=alt.Color('neighborhood_type:N', scale=color_scale),
        column='neighborhood_type:N',
        tooltip=['predicted_class', 'count()']
    ).transform_filter(alt.datum.neighborhood_type != 'instance')
    .transform_filter(brush)  # filter the data based on the brush selection
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
    ).transform_filter(brush))
    attributeCharts = []
    for attribute in attributes:
        attributeBarChart = marginalCharts.encode(
            x=alt.X(attribute, title=attribute)
            .bin(maxbins=20),
        ).properties(
            title=attribute,
        )
        attributeCharts.append(attributeBarChart)
    neighbsCharts = alt.vconcat(*attributeCharts)
    dashboard = (alt.vconcat(chartUMAP, chartClasses, neighbsCharts))
    return dashboard


@vessels_router.get("/descriptor", tags=["Vessels"])
async def descriptor():
    ds = TabularDataset(data=df_vessels, class_name='class N',categorial_columns=['class N'])

    return ds.descriptor

@vessels_router.post("/classify", tags=["Vessels"])
async def classify_vessel(vessel_event: VesselEvent):
    logger.info(f"Received event: {vessel_event}")

    instance = vessel_event.to_list()
    predicted_class = vessels_model.predict([instance])

    return {
        "predicted_class": predicted_class[0],
        "instance": instance
    }

@vessels_router.post("/neighborhood", tags=["Vessels"])
async def neighborhood(neigh_request: VesselRequest):
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
            "neighborhood_types": [
                "train", "custom"
            ]
        }
        ```
        The possible terms for the `neighborhood_types` fields are one of `['train', 'random', 'custom', 'genetic', 'custom_genetic']`
    """
    logger.info(f"Received event: {neigh_request.vessel_event}")
    logger.info(f"Number of samples: {neigh_request.num_samples}")
    logger.info(f"Neighborhood types: {neigh_request.neighborhood_types}")

    df_neighbs = await compute_neighborhoods(neigh_request)

    # return df_neighbs as a csv data
    stream = io.StringIO()
    df_neighbs.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=neighborhood.csv"

    return response

@vessels_router.post("/neighborhood/visualization", tags=["Vessels"])
async def neighborhood_visualization(neigh_request: VesselRequest):
    logger.info(f"Received event: {neigh_request.vessel_event}")
    logger.info(f"Number of size of neighborhood: {neigh_request.num_samples}")
    neighborhood_types_str = neigh_request.neighborhood_types
    logger.info(f"Neighborhood types: {neighborhood_types_str}")

    df_neighbs = await compute_neighborhoods(neigh_request)
    view = dataframe_to_vega(df_neighbs).to_json()

    return Response(content=view, media_type="application/json")



async def compute_neighborhoods(neigh_request):
    neighborhood_types_str = neigh_request.neighborhood_types
    instance_event = np.array(neigh_request.vessel_event.to_list())
    # make a deep copy of the instance
    input_instance = np.copy(instance_event)
    predicted_class = vessels_model.predict([instance_event])
    logger.info(f"Predicted class: {predicted_class}")
    neighbs = generate_neighborhoods(instance_event, vessels_model, df_vessels, vessels_data_train, vessels_target_train,
                                     neigh_request.num_samples, neighborhood_types_str)
    # create an empty data frame to aggregate the neighborhoods
    df_neighbs = pd.DataFrame([instance_event], columns=df_vessels.columns[:-1])
    df_neighbs['predicted_class'] = predicted_class
    df_neighbs['neighborhood_type'] = 'instance'
    for i, neighb in enumerate(neighbs):
        label = neighb[0]
        logger.info(f"Processing neighborhood type: {label}")
        neighb_classes = vessels_model.predict(neighb[1])
        # create a dataframe containing all the columns of neighb, plus and the predicted class and the string label
        neighb_df = pd.DataFrame(neighb[1], columns=df_vessels.columns[:-1])
        if label == 'train':
            neighb_df['predicted_class'] = vessels_target_train
        else:
            neighb_df['predicted_class'] = neighb_classes
        neighb_df['neighborhood_type'] = neighborhood_types_str[i]

        df_neighbs = pd.concat([df_neighbs, neighb_df], ignore_index=True)
    # apply UMAP to the neighborhood data. Only to the features contained in the original df_vessels
    embedding = reducer.transform(df_neighbs[df_vessels.columns[:-1]])
    df_neighbs['umap1'] = embedding[:, 0]
    df_neighbs['umap2'] = embedding[:, 1]
    return df_neighbs


@vessels_router.post("/explain", tags=["Vessels"])
async def explain(request:VesselRequest):
    logger.info(f"Received event: {request.vessel_event}")
    logger.info(f"Number of size of neighborhood: {request.num_samples}")
    neighborhood_types_str = request.neighborhood_types
    logger.info(f"Neighborhood types: {neighborhood_types_str}")

    instance_event = np.array(request.vessel_event.to_list())

    predicted_class_event = vessels_model.predict([instance_event])
    logger.info(f"Predicted class: {predicted_class_event}")

    ds = TabularDataset(data=df_vessels, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(vessels_model)
    surrogate = DecisionTreeSurrogate()

    generators = neighborhood_type_to_generators(neighborhood_types_str, bbox, ds, encoder, vessels_data_train, vessels_target_train)

    explanations = {}
    for (n, gen) in generators:
        spec_lore = Lore(bbox, ds, encoder, gen, surrogate)
        explanation = spec_lore.explain(instance_event, num_instances=request.num_samples)
        # convert explanation to json string using json.dumps
        # rule = covid_rule_to_dict(explanation['rule'])
        # crRules = [covid_rule_to_dict(cr) for cr in explanation['counterfactuals']]
        explanations[n] = explanation


    return {
        'instance': instance_event.tolist(),
        'predicted_class': predicted_class_event[0],
        'explanations': explanations
    }
