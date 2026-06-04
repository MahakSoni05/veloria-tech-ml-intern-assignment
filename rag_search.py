import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# loading csv file
df = pd.read_csv("match_data.csv")

# converting rows into sentences
match_sentences = []

for index, row in df.iterrows():

    sentence = (
        f"{row['Team 1']} vs {row['Team 2']} at {row['Venue']} "
        f"on {row['Match Date']}. "
        f"{row['Winner']} won. "
        f"Top scorer: {row['Top Scorer']} "
        f"with {row['Top Score']} runs."
    )

    match_sentences.append(sentence)

# loading embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# creating embeddings
embeddings = model.encode(match_sentences)

# user query
query = "show me matches where australia won"

# converting query into embedding
query_embedding = model.encode([query])

# finding similarity
similarity_scores = cosine_similarity(query_embedding, embeddings)

# getting top 3 matches
top_matches = similarity_scores[0].argsort()[-3:][::-1]

print("\nMost Relevant Matches:\n")

for i in top_matches:
    print(match_sentences[i])
    print()