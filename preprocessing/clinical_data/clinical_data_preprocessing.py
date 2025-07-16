import pandas as pd


def load_and_preprocess(path):
    """
    Loads tabular data of clinical records and preprocess it.
    """

    # Load data
    raw_data = pd.read_csv('data/clinical_data.csv')

    df = raw_data.copy(deep=True)

    # Impute missing data
    df.loc[df['PatientID'] == 'LUNG1-085', 'clinical.T.Stage'] = 1.0
    df.loc[df['PatientID'] == 'LUNG1-272', 'Overall.Stage'] = 'llb'
    df['age'] = df['age'].fillna(df['age'].mean())
    df['Histology'] = df['Histology'].fillna(df['Histology'].mode())

    # One-hot encode categorical data
    df = pd.concat([raw_data, pd.get_dummies(raw_data['Overall.Stage'], prefix='stage'),
                    pd.get_dummies(raw_data['Histology'], prefix='histology'),
                    pd.get_dummies(raw_data['clinical.T.Stage'], prefix='T.stage'),
                    pd.get_dummies(raw_data['Clinical.N.Stage'], prefix='N.stage'),
                    pd.get_dummies(raw_data['Clinical.M.Stage'], prefix='M.stage')],
                   axis=1).drop(
        columns=['Overall.Stage', 'Histology', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage'])
    df['is_male'] = df['gender'] == 'male'

    # Normalise data
    df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
    return df
