import pandas as pd
import numpy as np


def load_and_preprocess(path):
    """
    Loads tabular data of clinical records and preprocess it.
    """
    raw_data = pd.read_csv('../../data/clinical_data.csv')

    df = pd.DataFrame()
    df['id'] = raw_data['Case ID']
    df['time_to_death'] = raw_data['Time to Death (days)']

    # Impute missing data
    df['age'] = raw_data['Age at Histological Diagnosis'].fillna(raw_data['Age at Histological Diagnosis'].mean())

    # One-hot encode categorical data
    df['is_male'] = raw_data['Gender'] == 'Male'
    df['is_alive'] = raw_data['Survival Status'] == 'Alive'
    df = pd.concat([df,
                    pd.get_dummies(raw_data['Pathological T stage'], prefix='path_T_stage'),
                    pd.get_dummies(raw_data['Pathological N stage'], prefix='path_N_stage'),
                    pd.get_dummies(raw_data['Pathological M stage'], prefix='path_M_stage'),
                    pd.get_dummies(raw_data['Histology '], prefix='histology'),
                    pd.get_dummies(raw_data['Smoking status'], prefix='smoking_status'),
                    pd.get_dummies(raw_data['Ethnicity'], prefix='ethnicity'),
                    pd.get_dummies(raw_data['Histopathological Grade'], prefix='hist_grade')], axis=1)

    # Normalise data
    cont_cols = ['age', 'time_to_death']
    df[cont_cols] = (df[cont_cols] - df[cont_cols].mean()) / np.sqrt(df[cont_cols].var())

    return df


if __name__ == '__main__':
    path = '../../data/clinical_data.csv'
    df = load_and_preprocess(path)
    pd.set_option('display.width', None)
    print(df.head())
