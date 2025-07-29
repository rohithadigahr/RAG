from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline
import time
import psutil
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/rag")
def rag_search(request: QueryRequest):
    start_time = time.time()

    # Memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Get results
    result = pipeline.retrieve(request.query, request.top_k)

    # Step 2: Combine context
    context_text = "\n".join([
        f"Topic: {doc['topic']}\nText: {doc['ki_text']}\nAlt Text: {doc['alt_ki_text']}"
        for doc in result
    ])
    print(time.time())
    # Step 3: Generate answer
    generated_answer = pipeline.generate_answer(request.query, context_text)

    # Memory after
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_used = round(mem_after - mem_before, 2)

    # Latency
    latency_ms = round((time.time() - start_time) * 1000, 2)/1000
    # Quality of retrieval (cosine similarity between query and top doc)
    query_embed = pipeline.embed_query(request.query)
    top_doc_embed = pipeline.embed_query(result)
    # Ensure embeddings are 2D

    if len(query_embed.shape) == 1:
        query_embed = np.expand_dims(query_embed, axis=0)
    if len(top_doc_embed.shape) == 1:
        top_doc_embed = np.expand_dims(top_doc_embed, axis=0)

    # Compute cosine similarity
    retrieval_quality = round(
        float(cosine_similarity(query_embed, top_doc_embed)[0][0]), 4
    )

    # Evaluate how close the generated response is to the retrieved documents
    result = pipeline.combine_context(result)
    similarity_score = pipeline.evaluate_generation_vs_retrieved(generated_answer, result, pipeline.embedder)


    return {
        "query": request.query,
        "top_k": request.top_k,
        "answer": generated_answer,
        "context": result,
        "latency_ms": latency_ms,
        "memory_used_mb": mem_used,
        "retrieval_quality": retrieval_quality,
        "generation_quality": similarity_score,
    }
