import pandas as pd

print("Starting cricket match data collection...")

matches = []

matches.append({
    "Match Date": "2024-06-01",
    "Team 1": "India",
    "Team 2": "Australia",
    "Venue": "Mumbai",
    "Winner": "India",
    "Top Scorer": "Virat Kohli"
})

df = pd.DataFrame(matches)

df.to_csv("match_data.csv", index=False)

print("CSV file created successfully!")