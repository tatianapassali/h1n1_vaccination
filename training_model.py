# Import neccesary packages
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from sklearn.metrics import accuracy_score, f1_score
from data_loader import preprocess_dataframe, split_and_normalize

# Read training data
train_data = pd.read_csv("data/training_set_features.csv")
# Read target labels
labels = pd.read_csv("data/training_set_labels.csv")

# Preprocces the data
data, labels = preprocess_dataframe(train_data, labels)

# Split the data into train and test and standarize their values
X_train, X_test, y_train, y_test = split_and_normalize(data, labels)

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
