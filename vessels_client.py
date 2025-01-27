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
    payload = {
        'vessel_event': {
            'SpeedMinimum': 0,
            'SpeedQ1': 0,
            'SpeedMedian': 0,
            'SpeedQ3': 0,
            'DistanceStartShapeCurvature': 0,
            'DistanceStartTrendAngle': 0,
            'DistStartTrendDevAmplitude': 0,
            'MaxDistPort': 0,
            'MinDistPort': 0,
        },
        'num_samples': 3000,
        'neighborhood_types': 7 # in binary format: 111
    }

    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
    # the response contains the data in CSV format. Store it as a DataFrame
    df = pd.read_csv(io.StringIO(response.text))

    # create a nominal colro scale for the neighborhood types
    color_scale = alt.Scale(domain=['train', 'random', 'custom', 'genetic', 'custom_genetic'],
                            range=['#dddddd', '#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4'])

    # create a chart of the projected points
    chartUMAP = alt.Chart(df).mark_point().encode(
        x='umap1:Q',
        y='umap2:Q',
        color=alt.Color('neighborhood_type:N', scale=color_scale),
        shape='predicted_class:N',
        tooltip=['predicted_class', 'neighborhood_type', 'SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']
    ).properties(
        width=800,
        height=800
    ).interactive()

    chartClasses = alt.Chart(df).mark_bar().encode(
        x='predicted_class:N',
        y='count()',
        color=alt.Color('neighborhood_type:N', scale=color_scale),
        column='neighborhood_type:N',
        tooltip=['predicted_class', 'count()']
    ).properties(
        width=200,
        height=200
    )



    alt.vconcat(chartUMAP, chartClasses).save('vessels.html')


if __name__ == '__main__':
    get_vessels_data()