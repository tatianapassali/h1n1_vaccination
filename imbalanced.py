from imblearn.over_sampling import BorderlineSMOTE
from imblearn.ensemble import EasyEnsemble
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def simple_model(X_train,y_train):
    

    # define the methods
    over = SMOTE( k_neighbors=7)
    under = RandomUnderSampler()
    
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    new_X_train, new_y_train = pipeline.fit_resample(X_train, y_train)
    
    return new_X_train,new_y_train

def ensemble_model(X_train,y_train):
    
    # define the methods
    over = BorderlineSMOTE( k_neighbors=3, kind= "borderline-1")
    under = EasyEnsemble(random_state=1)

    steps = [('o', over), ('u', under)]

    pipeline = Pipeline(steps=steps)

    # transform the dataset
    new_X_train, new_y_train = pipeline.fit_resample(X_train, y_train)    
    
    return new_X_train[0],new_y_train[0]
