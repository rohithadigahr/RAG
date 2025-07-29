import pandas as pd
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from huggingface_hub import login
import requests
from sklearn.metrics.pairwise import cosine_similarity


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index("index/faiss.index")
        self.metadata = pd.read_csv("index/doc_metadata.csv")
        
    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for idx in indices[0]:
            results.append({
                "topic": self.metadata.iloc[idx]['ki_topic'],
                "ki_text": self.metadata.iloc[idx]['ki_text'],
                "alt_ki_text": self.metadata.iloc[idx]['alt_ki_text']
            })

        return results

    def generate_answer(self, query: str, context: str) -> str:

        prompt = (
        f""" Based on the following documentation, write a step-by-step guide to answer the user's question using only the documentation.
            Do not provide any additional information on your own.

            ### Documentation:
            {context}

            ### Question:
            Question: {query}

            ### Answer:Should have all the steps mentioned in the documentation
            
            Important note :
            1.Do not provide any other details outside the context.The steps should be only based on documentation provided."""
        )

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        })

        if response.status_code == 200:
            print(response.json()["response"])
        else:
            return "Failed to get response from tinyllama."

        return response.json()['response']

    def embed_query(self, text: str):
        return self.embedder.encode(text, normalize_embeddings=True)



    def evaluate_generation_vs_retrieved(self,generated_answer: str, context: str, embedder) -> float:
        # Ensure both are strings, not tuples
        answer_embed = embedder.encode([generated_answer], normalize_embeddings=True)
        context_embed = embedder.encode([context], normalize_embeddings=True)
        similarity_score = cosine_similarity(answer_embed, context_embed)[0][0]
        return round(float(similarity_score), 4)

    def combine_context(self,retrieved_docs: list) -> str:
        return " ".join(doc.get('ki_text', '') + " " + doc.get('alt_ki_text', '') for doc in retrieved_docs)


