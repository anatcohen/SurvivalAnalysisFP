import pandas as pd
import numpy as np


def load_split_and_preprocess(path: str, train_prop: float = 0.6, val_prop: float = 0.2, test_prop: float = 0.2):
    """
    Loads tabular data of clinical records and preprocess it.

    Parameters
    ----------
    path: str
        path to clinical data csv.
    train_prop: float, optional
        Proportion of train set to split by. Default value is 60%.
    val_prop: float, optional
        Proportion of validation set to split by. Default value is 20%.
    test_prop: float, optional
        Proportion of test set to split by. Default value is 20%.
    """
    # Verify train-val-test ratio
    assert train_prop+val_prop+test_prop == 1, 'train-val-test ratio must sum to one'

    # Load data
    raw_data = pd.read_csv(path)

    # Fill in specific missing data
    raw_data.loc[raw_data['PatientID'] == 'LUNG1-085', 'clinical.T.Stage'] = 1.0
    raw_data.loc[raw_data['PatientID'] == 'LUNG1-272', 'Overall.Stage'] = 'llb'

    # Split into train
    rows = raw_data.shape[0]
    train_ind = np.random.choice(rows, size=np.round(rows*train_prop).astype(int), replace=False)
    # Split remaining into val-test sets
    non_train_rows = rows-len(train_ind)
    test_prop = test_prop/(1-train_prop)
    test_ind = np.random.choice(non_train_rows, size=np.round(non_train_rows*test_prop).astype(int), replace=False)

    # Fill in missing values according to train set
    age_fill = raw_data.loc[train_ind, 'age'].mean()
    hist_fill = raw_data.loc[train_ind, 'Histology'].mode().iloc[0]

    raw_data['age'] = raw_data['age'].fillna(age_fill)
    raw_data['Histology'] = raw_data['Histology'].fillna(hist_fill)

    # One-hot encode categorical data
    df = pd.concat([raw_data, pd.get_dummies(raw_data['Overall.Stage'], prefix='stage', dtype=int),
                    pd.get_dummies(raw_data['Histology'], prefix='histology', dtype=int),
                    pd.get_dummies(raw_data['clinical.T.Stage'], prefix='T.stage', dtype=int),
                    pd.get_dummies(raw_data['Clinical.N.Stage'], prefix='N.stage', dtype=int),
                    pd.get_dummies(raw_data['Clinical.M.Stage'], prefix='M.stage', dtype=int)],
                   axis=1)

    df['is_male'] = (raw_data['gender'] == 'male').astype(int)
    df.drop(columns=['Overall.Stage', 'Histology', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'gender'],
            inplace=True)

    # Scale according to train set
    df['age'] = (df['age'] - df.loc[train_ind, 'age'].min())/(df.loc[train_ind, 'age'].max() - df.loc[train_ind, 'age'].min())

    # Get each split
    train_df = df.iloc[train_ind]
    val_df = df.drop(index=train_ind).reset_index().drop(index=test_ind, columns='index')
    test_df = (df.drop(index=train_ind)).reset_index().iloc[test_ind].drop(columns='index')

    return train_df, val_df, test_df

if __name__ == '__main__':
    train, val, test = load_split_and_preprocess('../../data/clinical_data.csv')
    train.to_csv('../../data/preprocessed_clinical_data_train.csv', index=False)
    val.to_csv('../../data/preprocessed_clinical_data_val.csv', index=False)
    test.to_csv('../../data/preprocessed_clinical_data_test.csv', index=False)

    df = pd.concat([train, val, test])
    print(df)
    nan_counts = df.isnull().sum()
    columns_with_nans = nan_counts[nan_counts > 0]

    if len(columns_with_nans) > 0:
        print("Columns with NaN values:")
        print(columns_with_nans)
    else:
        print("No NaN values found!")
