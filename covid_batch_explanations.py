import pandas as pd
import numpy as np

from lore_sa.dataset import TabularDataset
from covid_router import explain, CovidEvent, CovidRequest, res

def rule_to_dict(rule, descriptor):
    """
    Converts a rule to a dictionary where each key is one of the features of the domain dataset and the value is the
    interval of definition of the rule.

    Parameters:
        explanation: a rule object from lore
        descriptor: a descriptor object from lore

    Returns:

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
    receives a dictionary of intervals and returns a string representation of the intervals
    The string output for each feature has the form: feature_name: {lower_bound, upper_bound}. The features are separated
    by a semicolon.
    """
    str_intervals = []
    for f in intervals:
        str_intervals.append(f"{f}: " + "{" + str(intervals[f][0]) + ": " + str(intervals[f][1]) + "}")
    return "; ".join(str_intervals)




async def main():
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
