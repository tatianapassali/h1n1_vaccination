# Import necessary packages

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imbalanced import ensemble_model, simple_model
from Interpret import Interpret, white_box
from data_loader import preprocess_dataframe, split_and_normalize
from utilities import prediction_evaluation

SIMPLE_IMBALANCE = 0
INTERPRET = 0


def main():
    # Read training data
    train_data = pd.read_csv("data/training_set_features.csv")
    # Read target labels
    labels = pd.read_csv("data/training_set_labels.csv")

    # TODO ------- Future work ------
    # temp = Preprocessor(train_data, labels).encode()
    # print(temp.named_transformers_["object"].categories_)
    # TODO ===========================

    # Preprocess the data
    data, labels = preprocess_dataframe(train_data, labels)

    # Run only once to create a global split for training and test saved in pickle
    # create_dataset_splits(data, labels)

    # Split the data into train and test
    x_train, x_test, y_train, y_test = split_and_normalize(data, labels)
    if SIMPLE_IMBALANCE:
        x_train, y_train = simple_model(x_train, y_train)
    else:
        x_train, y_train = ensemble_model(x_train, y_train)
    # Time for Active Learning
    # Implement each strategy
    # strategy1_examples = create_and_implement_strategy("QueryInstanceUncertainty", data, labels, queries)
    # strategy2_examples = create_and_implement_strategy("QueryInstanceRandom", data, labels, queries)
    # strategy3_examples = create_and_implement_strategy("QueryInstanceQBC", data, labels, queries)
    # Plot learning curves
    # plot_learning_curves(strategy1_examples, strategy2_examples, strategy3_examples)

    # White box interpretation
    drop = ['h1n1_concern', 'income_poverty', 'household_children', 'household_adults', 'behavioral_antiviral_meds',
            'race', 'child_under_6_months', 'chronic_med_condition', 'education', 'behavioral_wash_hands',
            'behavioral_outside_home', 'behavioral_touch_face', 'behavioral_face_mask', 'behavioral_avoidance',
            'rent_or_own', 'sex', 'employment_status', 'hhs_geo_region', 'behavioral_large_gatherings']
    # x_train = x_train.drop(drop, axis=1)
    # x_test = x_test.drop(drop, axis=1)
    # white_box(x_train, y_train, x_test, y_test)
    # return


    # Black box interpretation
    black_box = Interpret(RandomForestClassifier(n_estimators=750, random_state=1, max_depth=5))
    # black_box = Interpret(LinearSVC())
    black_box.fit(x_train, y_train)
    prediction = black_box.predict(x_test, y_test)
    # Evaluate the model
    prediction_evaluation(prediction=prediction, y_test=y_test)

    # Interpret the model
    if not INTERPRET:
        black_box.feature_importance()
    else:
        black_box.shap_interpret()
    # black_box.lime(num_features=len(x_train.columns))
    # black_box.surrogate(LogisticRegression())


if __name__ == "__main__":
    main()

# EOF
