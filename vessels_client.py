# This script contains statements to connect to the Vessels API and retrieve data from it.
import io

import pandas as pd
import requests
import json
import altair as alt

alt.data_transformers.enable('default', max_rows=None)


# Function to get the data from the Vessels API
# Prepare and send a POST request to the Vessels API
def get_vessels_data():
    url = "http://localhost:10000/neighborhood"
    headers = {
        'accept': "application/json",
        'Content-Type': "application/json"
    }
    # prepare the payload for the POST request
    vessel_event_values = [0.05, 0.08, 0.12, 0.16, 52.35, 0, 0.01, 0.32, 0.32] # row id 3, class N = 6

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
        'num_samples': 3000,
        # 'neighborhood_types': 7 # in binary format: 111
        'neighborhood_types': 23 # in binary format: 10111
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
        tooltip=['predicted_class', 'neighborhood_type', 'SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']
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

    grouped = df.groupby(['neighborhood_type'])
    attributes = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']
    neighbsCharts = []
    for key, group in grouped:
        print(key)
        if key[0] != 'instance':
            barChart = alt.Chart(group).mark_bar().encode(
                y = alt.Y('count()', title=''),
                color=alt.Color('neighborhood_type:N', scale=color_scale),
            )
            multiCharts = []
            for attribute in attributes:
                print(attribute)
                attributeBarChart = barChart.encode(
                    x = alt.X(attribute, title=attribute)
                        .bin(maxbins=20)
                ).properties(
                    title=attribute + ' - ' + key[0],
                    height=100,
                    width=200,
                ).transform_filter(brush)
                multiCharts.append(attributeBarChart)

            neighbTypeCharts = alt.vconcat(*multiCharts)
            neighbsCharts.append(neighbTypeCharts)



    alt.vconcat(chartUMAP, chartClasses, alt.hconcat(*neighbsCharts)).save('vessels.html')


if __name__ == '__main__':
    get_vessels_data()