# Import neccesary packages
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


# Preprocees data of the dataframe (drop useless columns, replace string values with categorical and fill empty records with dummy value)
def preprocess_dataframe(dataframe, labels):
    dataframe = dataframe.drop(
        ['respondent_id', 'doctor_recc_seasonal', 'opinion_seas_vacc_effective', 'opinion_seas_risk',
         'opinion_seas_sick_from_vacc', 'employment_industry', 'employment_occupation'], axis=1)
    labels = labels.drop(['respondent_id', 'seasonal_vaccine', ], axis=1)
    dataframe.age_group.replace(
        ['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years'], [0, 1, 2, 3, 4],
        inplace=True)
    dataframe.education.replace(['< 12 Years', '12 Years', 'Some College', 'College Graduate'],
                                [0, 1, 2, 3], inplace=True)
    dataframe.race.replace(['White', 'Black', 'Hispanic', 'Other or Multiple'], [0, 1, 2, 3],
                           inplace=True)
    dataframe.sex.replace(['Male', 'Female'], [0, 1], inplace=True)
    dataframe.income_poverty.replace(['<= $75,000, Above Poverty', '> $75,000', 'Below Poverty'],
                                     [0, 1, 2], inplace=True)
    dataframe.marital_status.replace(['Married', 'Not Married'], [0, 1], inplace=True)
    dataframe.rent_or_own.replace(['Rent', 'Own'], [0, 1], inplace=True)
    dataframe.employment_status.replace(['Employed', 'Not in Labor Force', 'Unemployed'],
                                        [0, 1, 2], inplace=True)
    dataframe.hhs_geo_region.replace(
        ['lzgpxyit', 'fpwskwrf', 'qufhixun', 'oxchjgsf', 'kbazzjca', 'bhuqouqj', 'mlyzmhmf', 'lrircsnp', 'atmpeygn',
         'dqpwygqj'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
    dataframe.census_msa.replace(['MSA, Not Principle  City', 'MSA, Principle City', 'Non-MSA'],
                                 [0, 1, 2], inplace=True)
    dataframe = dataframe.fillna(20)  # Replace nan with dummy value

    return dataframe, labels


def split_and_normalize(data, labels):
    # Split data into train and test set ( ratio 0.2 test - 0.8 train)
    with open("dataset", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # # Compute mean and std
    # mean = np.mean(X_train, axis=0)
    # std = np.std(X_train, axis=0)
    #
    # # Standardize data
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std
    X_train = reverse(X_train)
    X_test = reverse(X_test)
    return X_train, X_test, pd.DataFrame(y_train, columns=['h1n1_vaccine']), pd.DataFrame(y_test, columns=['h1n1_vaccine'])


def reverse(x):
    return pd.DataFrame(x, columns=['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
                                   'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
                                   'behavioral_large_gatherings', 'behavioral_outside_home',
                                   'behavioral_touch_face', 'doctor_recc_h1n1', 'chronic_med_condition',
                                   'child_under_6_months', 'health_worker', 'health_insurance',
                                   'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                                   'opinion_h1n1_sick_from_vacc', 'age_group', 'education', 'race', 'sex',
                                   'income_poverty', 'marital_status', 'rent_or_own', 'employment_status',
                                   'hhs_geo_region', 'census_msa', 'household_adults',
                                   'household_children'])
