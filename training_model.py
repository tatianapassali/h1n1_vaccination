# Import necessary packages
import pandas as pd
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier

from Models import Interpret
from data_loader import preprocess_dataframe, split_and_normalize
from utilities import prediction_evaluation


def tatianas_cute_agent():
    # Create neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=29, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Add dropout layer to avoid overfitting
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=16)
    y_pred = model.predict(X_test)

    # print(y_pred)
    # print(min(y_pred), max(y_pred))
    # Predict label as 1 if prediction is greater than 0.5
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Accuracy is: ", 100 * accuracy)
    print("F1 is: ", f1)


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

    # Split the data into train and test and standardize their values
    x_train, x_test, y_train, y_test = split_and_normalize(data, labels)

    black_box = Interpret(RandomForestClassifier(n_estimators=750, random_state=1, max_depth=5))
    # black_box = Interpret(LinearSVC())
    black_box.fit(x_train, y_train)
    prediction = black_box.predict(x_test)
    # Evaluate the model
    prediction_evaluation(prediction=prediction, y_test=y_test)

    # Interpret the model
    black_box.feature_importance(y_test=y_test)
    black_box.shap_interpret(y_test=y_test)


if __name__ == "__main__":
    main()

# EOF
