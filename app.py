import pandas as pd
from openai.embeddings_utils import cosine_similarity

# Load the dataset (assuming it's a CSV)
df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Preprocess the data (remove punctuation, lowercase, etc.)
df['Question'] = df['Question'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True)

# Initialize OpenAI API key

# Function to get embedding
def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Pre-calculate embeddings for all questions in the dataset
df['Question_Embedding'] = df['Question'].apply(get_embedding)

# Save the DataFrame with embeddings and similarities to a CSV file
df.to_csv("qa_dataset_with_embeddings.csv", index=False)  # index=False to avoid saving row numbers
