"""
COVID-19 Batch Explanation Processing

This module provides batch processing capabilities for generating COVID-19 risk
explanations. It's designed to process multiple instances from a CSV file and
generate explanations using various neighborhood generation strategies.

The module is useful for:
- Evaluating explanation quality across many instances
- Comparing different neighborhood generation strategies
- Creating datasets of explanations for analysis
- Benchmarking explanation generation performance

Functions:
    rule_to_dict: Convert LORE rule to dictionary of feature intervals
    intervals_to_str: Convert interval dictionary to string representation
    main: Async main function for batch processing

Usage:
    python covid_batch_explanations.py
    
The script processes instances from 'datasets/selected_train_instances_15.csv'
and generates explanations using multiple neighborhood types.
"""

import pandas as pd
import numpy as np

from lore_sa.dataset import TabularDataset
from covid_router import explain, CovidEvent, CovidRequest, res

def rule_to_dict(rule, descriptor):
    """
    Convert a LORE rule to a dictionary of feature intervals.
    
    Transforms a rule from the LORE explanation into a dictionary where
    each key is a feature name and the value is a [min, max] interval
    defining the rule's conditions for that feature.
    
    Args:
        rule: Rule object from LORE with 'premises' containing conditions
        descriptor: Dataset descriptor with categorical and numeric features
    
    Returns:
        dict: Mapping of feature names to [lower_bound, upper_bound] intervals
              Unbounded intervals use -np.inf or np.inf
              
    Example:
        >>> rule = {'premises': [{'attr': 'age', 'op': '>', 'val': 30}, 
        ...                      {'attr': 'age', 'op': '<=', 'val': 50}]}
        >>> result = rule_to_dict(rule, descriptor)
        >>> result['age']  # [30, 50]
    """
    descr_features = []
    for cf in descriptor['categorical']:
        descr_features.append({cf: descriptor['categorical'][cf]})
    for nf in descriptor['numeric']:
        descr_features.append({nf: descriptor['numeric'][nf]})

    intervals = {}
    for i, f in enumerate(descr_features):
        # the feature f should contain a single key
        fkey = list(f.keys())[0]
        boundaries =[-np.inf, np.inf]
        for j, iv in enumerate(rule['premises']):
            if fkey == iv['attr']:
                if iv['op'] == '<=' and iv['val'] < boundaries[1]:
                    boundaries[1] = iv['val']
                elif iv['op'] == '>' and iv['val'] > boundaries[0]:
                    boundaries[0] = iv['val']
        intervals[fkey] = boundaries

    return intervals

def intervals_to_str(intervals):
    """
    Convert feature intervals dictionary to string representation.
    
    Creates a human-readable string representation of rule intervals
    in the format: "feature1: {lower, upper}; feature2: {lower, upper}"
    
    Args:
        intervals: Dictionary mapping feature names to [min, max] intervals
    
    Returns:
        str: Semicolon-separated string of interval representations
        
    Example:
        >>> intervals = {'age': [30, 50], 'income': [40000, 80000]}
        >>> intervals_to_str(intervals)
        'age: {30: 50}; income: {40000: 80000}'
    """
    str_intervals = []
    for f in intervals:
        str_intervals.append(f"{f}: " + "{" + str(intervals[f][0]) + ": " + str(intervals[f][1]) + "}")
    return "; ".join(str_intervals)




async def main():
    """
    Main batch processing function for COVID-19 explanations.
    
    Processes multiple COVID-19 instances from a CSV file and generates
    explanations using various neighborhood generation strategies. The
    function demonstrates how to:
    1. Load instances from CSV
    2. Create CovidEvent and CovidRequest objects
    3. Generate explanations for multiple neighborhood types
    4. Extract and format rule intervals
    
    The output includes:
    - Instance ID
    - Predicted class
    - Neighborhood type
    - Rule intervals for each feature
    
    Note: Currently processes only the first instance (break after first).
          Remove the break statement to process all instances.
    """
    df = pd.read_csv('datasets/selected_train_instances_15.csv', sep=';')

    print(df.columns)
    print(df.shape)

    output_csv = 'output_covid_batch_15.csv'

    for i, row in df.iterrows():

        ce = CovidEvent(
            week6_covid=row['Week6_Covid'],
            week5_covid=row['Week5_Covid'],
            week4_covid=row['Week4_Covid'],
            week3_covid=row['Week3_Covid'],
            week2_covid=row['Week2_Covid'],
            week6_mobility=row['Week6_Mobility'],
            week5_mobility=row['Week5_Mobility'],
            week4_mobility=row['Week4_Mobility'],
            week3_mobility=row['Week3_Mobility'],
            week2_mobility=row['Week2_Mobility'],
            week1_mobility=row['Week1_Mobility'],
            days_passed=row['Days_passed']
        )

        cr = CovidRequest(
            event=ce,
            num_samples=2000,
            neighborhood_types=['random', 'custom', 'genetic', 'gpt', 'baseline']
        )

        ds = TabularDataset(data=res, class_name='Class_label', categorial_columns=['Class_label'])

        response = await explain(cr)
        explanations = response['explanations']

        for n in explanations:
            # output
            # instance id, predicted class, rule_id(R0, C1, C2, ...), rule intervals
            intervals = rule_to_dict(explanations[n]['rule'], ds.descriptor)
            print(f"{i}, {explanations[n]['predicted_class']}, {n}, {intervals}")

        break

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
