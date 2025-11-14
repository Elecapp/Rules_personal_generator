"""
Vessel Movement Batch Explanation Processing

This module provides batch processing capabilities for generating vessel movement
pattern explanations. It processes multiple vessel trajectory instances from a CSV
file and generates explanations using various neighborhood generation strategies.

The module is useful for:
- Large-scale evaluation of explanation quality
- Comparing different neighborhood generation methods
- Creating comprehensive explanation datasets
- Benchmarking performance across vessel classes

Functions:
    main: Async main function for batch vessel explanation processing

Usage:
    python vessels_batch_explanations.py
    
The script:
1. Loads vessel instances from 'datasets/instances_6x20_data_encoded.csv'
2. Renames columns to match expected feature names
3. Generates explanations for each instance using multiple neighborhood types
4. Outputs results to CSV with rule intervals and counterfactuals

Output format:
    instance_id, rule_type, neighborhood_type, intervals, predicted_class
    - rule_type: R0 (main rule) or C1, C2... (counterfactual rules)
    - intervals: Feature value ranges defining the rule
"""

import asyncio

import pandas as pd

from lore_sa.dataset import TabularDataset
from vessels_router import explain, VesselEvent, VesselRequest, rule_to_dict, intervals_to_str, df_vessels, descriptor


async def main():
    """
    Main batch processing function for vessel movement explanations.
    
    Processes all vessel instances from the input CSV and generates explanations
    using multiple neighborhood generation strategies. For each instance:
    1. Creates a VesselEvent with trajectory features
    2. Requests explanations using different neighborhood types
    3. Extracts main rule (R0) and counterfactual rules (C1, C2, ...)
    4. Writes results to output CSV
    
    The function demonstrates:
    - Handling column name mapping from raw data
    - Creating VesselEvent and VesselRequest objects
    - Processing explanations for multiple neighborhood types
    - Extracting both factual and counterfactual rules
    - Writing structured output for analysis
    
    Neighborhood types used:
    - random: Random generation
    - custom: Feature importance-based
    - genetic: Genetic algorithm
    - custom_genetic: Genetic with feature importance
    - baseline: Training data neighbors
    - llm: LLM-inspired with constraints
    """
    df = pd.read_csv('datasets/instances_6x20_data_encoded.csv')


    rename_dict = {
        'Mean speed in 5 minutes (km/h): minimum': 'SpeedMinimum',
        'Mean speed in 5 minutes (km/h): quartile 1': 'SpeedQ1',
        'Mean speed in 5 minutes (km/h): median': 'SpeedMedian',
        'Mean speed in 5 minutes (km/h): quartile 3': 'SpeedQ3',
        'Distance from the start  shape curvature': 'DistanceStartShapeCurvature',
        'Distance from the start trend line angle': 'DistanceStartTrendAngle',
        'Amplitude of deviations from the trend line of distance from the start': 'DistStartTrendDevAmplitude',
        'Max distance to nearest port (km)': 'MaxDistPort',
        'Min distance to nearest port (km)': 'MinDistPort',
        'Log10 of distance from the start: shape curvature': 'Log10Curvature',
        'Log10 of amplitude of deviations from the trend line of distance from the start': 'Log10DistStartTrendDevAmplitude',
        'Log10 of min distance to nearest port': 'Log10MinDistPort'
    }

    # Rename the columns
    df = df.rename(columns=rename_dict)
    print(df.columns)
    print(df.shape)

    output_csv = 'output_vessels_batch_120_logAttribute_n2000_all_neighbs.csv'

    for i, row in df.iterrows():

        ve = VesselEvent(
            SpeedMinimum = row['SpeedMinimum'],
            SpeedQ1 = row['SpeedQ1'],
            SpeedMedian = row['SpeedMedian'],
            SpeedQ3 = row['SpeedQ3'],
            Log10Curvature = row['Log10Curvature'],
            DistanceStartTrendAngle = row['DistanceStartTrendAngle'],
            Log10DistStartTrendDevAmplitude = row['Log10DistStartTrendDevAmplitude'],
            MaxDistPort = row['MaxDistPort'],
            Log10MinDistPort = row['MinDistPort']
        )

        vr = VesselRequest(
            vessel_event = ve,
            num_samples = 500,
            neighborhood_types = ['random', 'custom', 'genetic', 'custom_genetic', 'baseline', 'llm']
        )

        ds = TabularDataset(data=df_vessels, class_name='class N', categorial_columns=['class N'])

        response = await explain(vr)
        explanations = response['explanations']

        with open(output_csv, mode='a', newline='') as f:
            for n in explanations:
                # output
                # instance id, predicted class, rule_id(R0, C1, C2, ...), rule intervals
                intervals = rule_to_dict(explanations[n]['rule'], ds.descriptor)
                f.write(f'{row["id"]},R0,{n},{intervals_to_str(intervals)},{explanations[n]["rule"]["consequence"]["val"]}\n')
                print(intervals_to_str(intervals))
                for j, cr in enumerate(explanations[n]['counterfactuals']):
                    intervals = rule_to_dict(cr, ds.descriptor)
                    f.write(f'{row["id"]},C{j+1},{n},{intervals_to_str(intervals)},{cr["consequence"]["val"]}\n')
        print(f'Instance {row["id"]} done {i+1}/{df.shape[0]}')

if __name__ == '__main__':
    asyncio.run(main())

