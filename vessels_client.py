# This script contains statements to connect to the Vessels API and retrieve data from it.
import io

import pandas as pd
import requests
import json
import altair as alt

from main_vessels import load_data_from_csv

alt.data_transformers.enable('default', max_rows=None)

attributes = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']

# Function to get the data from the Vessels API
# Prepare and send a POST request to the Vessels API
def create_instance_visualization(vessel_event_values: list, filename: str = 'instance_visualization.html', num_samples: int = 1000, neighborhood_types: int = 31):


    url = "http://localhost:10000/vessels/neighborhood"
    headers = {
        'accept': "application/json",
        'Content-Type': "application/json"
    }
    # prepare the payload for the POST request
    payload = {
        'vessel_event': {
            'SpeedMinimum': vessel_event_values[0],
            'SpeedQ1': vessel_event_values[1],
            'SpeedMedian': vessel_event_values[2],
            'SpeedQ3': vessel_event_values[3],
            'DistanceStartShapeCurvature': vessel_event_values[4],
            'DistanceStartTrendAngle': vessel_event_values[5],
            'DistStartTrendDevAmplitude': vessel_event_values[6],
            'MaxDistPort': vessel_event_values[7],
            'MinDistPort': vessel_event_values[8]
        },
        'num_samples': num_samples,
        'neighborhood_types': neighborhood_types
    }

    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
    # the response contains the data in CSV format. Store it as a DataFrame
    df = pd.read_csv(io.StringIO(response.text))
    print(df.columns)

    # create a nominal colro scale for the neighborhood types
    color_scale = alt.Scale(domain=['instance', 'train', 'random', 'custom', 'genetic', 'custom_genetic'],
                            range=['#333333', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'])

    # create a chart of the projected points
    brush = alt.selection_interval(
        on="[pointerdown[event.altKey], pointerup] > pointermove",
        name='brush'
    )

    chartUMAP = alt.Chart(df).mark_point().encode(
        x='umap1:Q',
        y='umap2:Q',
        color=alt.when(brush).then(alt.Color('neighborhood_type:N', scale=color_scale)).otherwise(alt.value('lightgray')),
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
    .transform_filter(brush) # filter the data based on the brush selection
    .properties(
        width=200,
        height=200
    ))

    marginalCharts = (alt.Chart(df).mark_bar().encode(
        y = alt.Y('count()', title=''),
        color=alt.Color('neighborhood_type:N', scale=color_scale),
        column=alt.Column('neighborhood_type:N', title=None)
    ).transform_filter(alt.datum.neighborhood_type != 'instance').properties(
        width=200,
        height=100
    ).transform_filter(brush))

    attributeCharts = []
    for attribute in attributes:
        attributeBarChart = marginalCharts.encode(
            x = alt.X(attribute, title=attribute)
                .bin(maxbins=20),
        ).properties(
            title=attribute,
        )
        attributeCharts.append(attributeBarChart)

    neighbsCharts = alt.vconcat(*attributeCharts)

    (alt.vconcat(chartUMAP, chartClasses, neighbsCharts)
     .save(filename))


if __name__ == '__main__':
    df_vessels = load_data_from_csv()
    instance_id = 126
    # select the row with id instance_id and get only the values of the attributes
    vessel_event_values = df_vessels.iloc[instance_id][attributes].values.tolist()



    # vessel_event_values = [0.05, 0.08, 0.12, 0.16, 52.35, 0, 0.01, 0.32, 0.32] # row id 3, class N = 6
    # vessel_event_values = [1.37, 4.05, 4.45, 5.24, 2.27, 0, 4.58, 25.92, 21.8] # row id 3, class N = 3
    # vessel_event_values = [13.57, 14.05, 14.64, 16.65, 1, 0.25, 1.7, 34.67, 23.6] # row id 3, class N = 1


    # NEIGHB_TRAIN =          0b00001
    # NEIGHB_RANDOM =         0b00010
    # NEIGHB_CUSTOM =         0b00100
    # NEIGHB_GENETIC =        0b01000
    # NEIGHB_CUSTOM_GENETIC = 0b10000
    size_neighb = 5000
    neigh_types = 0b01001
    filename = f'vessels_{instance_id}_{size_neighb}_{neigh_types}_neighbs_.html'
    create_instance_visualization(vessel_event_values, filename, num_samples=size_neighb, neighborhood_types=neigh_types)