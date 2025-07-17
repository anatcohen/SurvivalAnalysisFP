import pandas as pd
import numpy as np

def load_and_preprocess(path):
    """
    Loads tabular data of clinical records and preprocess it.
    """

    # Load data
    raw_data = pd.read_csv(path)

    df = raw_data.copy(deep=True)

    # Impute missing data
    df.loc[df['PatientID'] == 'LUNG1-085', 'clinical.T.Stage'] = 1.0
    df.loc[df['PatientID'] == 'LUNG1-272', 'Overall.Stage'] = 'llb'
    df['age'] = df['age'].fillna(df['age'].mean())
    df['Histology'] = df['Histology'].fillna(df['Histology'].mode())

    # One-hot encode categorical data
    df = pd.concat([df, pd.get_dummies(raw_data['Overall.Stage'], prefix='stage', dtype=int),
                    pd.get_dummies(raw_data['Histology'], prefix='histology', dtype=int),
                    pd.get_dummies(raw_data['clinical.T.Stage'], prefix='T.stage', dtype=int),
                    pd.get_dummies(raw_data['Clinical.N.Stage'], prefix='N.stage', dtype=int),
                    pd.get_dummies(raw_data['Clinical.M.Stage'], prefix='M.stage', dtype=int)],
                   axis=1).drop(
        columns=['Overall.Stage', 'Histology', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'gender'])
    df['is_male'] = (raw_data['gender'] == 'male').astype(int)

    return df

if __name__ == '__main__':
    df = load_and_preprocess('../../data/clinical_data.csv')
    df.to_csv('../../data/preprocessed_clinical_data.csv', index=False)

    print(df)
    nan_counts = df.isnull().sum()
    columns_with_nans = nan_counts[nan_counts > 0]

    if len(columns_with_nans) > 0:
        print("Columns with NaN values:")
        print(columns_with_nans)
    else:
        print("No NaN values found!")
