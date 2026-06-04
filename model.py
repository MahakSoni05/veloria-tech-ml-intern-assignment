import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# reading csv file
data = pd.read_csv("match_data.csv")

print("Match Data")
print(data)


# removing empty rows
data = data.dropna()


# converting text into numbers
label_encoder = LabelEncoder()

data["Team 1"] = label_encoder.fit_transform(data["Team 1"])

data["Team 2"] = label_encoder.fit_transform(data["Team 2"])

data["Venue"] = label_encoder.fit_transform(data["Venue"])

data["Winner"] = label_encoder.fit_transform(data["Winner"])


# input features
X = data[["Team 1", "Team 2", "Venue"]]

# output
y = data["Winner"]


# splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# creating model
model = RandomForestClassifier()

# training model
model.fit(X_train, y_train)


# prediction
y_pred = model.predict(X_test)


# accuracy
accuracy = accuracy_score(y_test, y_pred)

# f1 score
f1 = f1_score(y_test, y_pred, average="weighted")

# confusion matrix
matrix = confusion_matrix(y_test, y_pred)


print("\nModel Trained Successfully")

print("\nAccuracy Score:")
print(accuracy)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(matrix)


# predicting a new match

new_match = pd.DataFrame(
    [[1, 0, 2]],
    columns=["Team 1", "Team 2", "Venue"]
)

prediction = model.predict(new_match)

print("\nPredicted Winner:")

if prediction[0] == 0:
    print("India")
else:
    print("Australia")