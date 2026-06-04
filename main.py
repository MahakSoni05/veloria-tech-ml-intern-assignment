import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# reading csv file
data = pd.read_csv("matches.csv")

print("Dataset Loaded")
print(data.head())


# encoding text data
team1_encoder = LabelEncoder()
team2_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
winner_encoder = LabelEncoder()

data["Team 1"] = team1_encoder.fit_transform(data["Team 1"])
data["Team 2"] = team2_encoder.fit_transform(data["Team 2"])
data["Venue"] = venue_encoder.fit_transform(data["Venue"])
data["Winner"] = winner_encoder.fit_transform(data["Winner"])


# selecting features
X = data[["Team 1", "Team 2", "Venue"]]

# target column
y = data["Winner"]


# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# training model
model = RandomForestClassifier()

model.fit(X_train, y_train)

print("Model Trained Successfully")


# checking accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy :", accuracy)


# prediction part
team1 = "India"
team2 = "Australia"
venue = "Sydney Cricket Ground Sydney"


# converting text to numbers
team1_value = team1_encoder.transform([team1])[0]
team2_value = team2_encoder.transform([team2])[0]
venue_value = venue_encoder.transform([venue])[0]


# predicting winner
result = model.predict([[team1_value, team2_value, venue_value]])

winner = winner_encoder.inverse_transform(result)

print("Predicted Winner :", winner[0])