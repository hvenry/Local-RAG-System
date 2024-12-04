import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def calculate_similarity_score(answer: str, context_documents: List[Document], embeddings: HuggingFaceEmbeddings) -> float:
    """
    Calculate the maximum cosine similarity score between the generated answer and the context documents.

    This function embeds the generated answer and the context documents, then calculates the cosine similarity
    between the answer embedding and each context document embedding. It returns the maximum similarity score.

    Args:
        answer (str): The generated answer text.
        context_documents (list): A list of context documents.
        embeddings (HuggingFaceEmbeddings): The embeddings model used to convert text to embeddings.

    Returns:
        float: The maximum cosine similarity score between the answer and the context documents.
    """
    
    # Embed the generated answer
    answer_embedding = embeddings.embed_documents([answer])[0]

    # Embed the context documents
    context_embeddings = [embeddings.embed_documents([doc.page_content])[0] for doc in context_documents]
    context_embeddings = np.array(context_embeddings)

    # Calculate cosine similarity between the answer and context documents
    similarities = cosine_similarity(np.array([answer_embedding]), context_embeddings)
    max_similarity_score = np.max(similarities)

    # Return the maximum calculated similarity score
    return max_similarity_score