# This script contains statements to connect to the Vessels API and retrieve data from it.
import io

import pandas as pd
import requests
import json
import altair as alt

import vessels_utils
from main_vessels import load_data_from_csv
from vessels_router import dataframe_to_vega

alt.data_transformers.enable('default', max_rows=None)

attributes = vessels_utils.vessels_features

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
    dashboard = dataframe_to_vega(df)
    dashboard.save(filename)


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