import pandas as pd
import numpy as np


def load_and_preprocess(path):
    """
    Loads tabular data of clinical records and preprocess it.
    """
    # Load data
    raw_data = pd.read_csv(path)

    # One-hot encode categorical data
    df = pd.concat([raw_data, pd.get_dummies(raw_data['Overall.Stage'], prefix='stage'),
                    pd.get_dummies(raw_data['Histology'], prefix='histology'),
                    pd.get_dummies(raw_data['clinical.T.Stage'], prefix='T.stage'),
                    pd.get_dummies(raw_data['Clinical.N.Stage'], prefix='N.stage'),
                    pd.get_dummies(raw_data['Clinical.M.Stage'], prefix='M.stage')],
                   axis=1)
    df['is_male'] = df['gender'] == 'male'
    df.drop(columns=['gender', 'Overall.Stage', 'Histology', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage'], inplace=True)

    # Impute missing data
    df['age'] = df['age'].fillna(df['age'].mean())

    # Normalise data
    cont_data = ['age']
    df[cont_data] = (df[cont_data] - df[cont_data].mean()) / df[cont_data].std()
    return df


if __name__ == '__main__':
    path = '../../data/clinical_data.csv'
    df = load_and_preprocess(path)
    pd.set_option('display.width', None)
    print(df.head())
