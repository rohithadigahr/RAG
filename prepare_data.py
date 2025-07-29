import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer

# Load CSV
df = pd.read_csv('data/synthetic_knowledge_items.csv')

# Drop any rows with missing values
df.dropna(subset=["ki_topic", "ki_text", "alt_ki_text"], inplace=True)

# Combine all 3 columns into one string per row for embedding
combined_texts = (
    df['ki_topic'].astype(str) + " | " +
    df['ki_text'].astype(str) + " | " +
    df['alt_ki_text'].astype(str)
).tolist()

# Create sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(combined_texts, show_progress_bar=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index and metadata
faiss.write_index(index, "index/faiss.index")
df.to_csv("index/doc_metadata.csv", index=False)

print("✅ FAISS index and metadata saved.")
