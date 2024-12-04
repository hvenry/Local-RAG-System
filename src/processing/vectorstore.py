import os
import numpy as np
import faiss
from typing import List, Tuple
from sklearn.preprocessing import normalize
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def load_documents_into_vectorstore(
    split_documents: List[Document], 
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    """
    Create a local vector database by converting split documents into vector representations.

    This function takes split documents, converts them into vector embeddings using a specified embedding model,
    and stores them in a FAISS (Facebook AI Similarity Search) vector database for efficient similarity search.

    Args:
        split_documents (list): A list of split document chunks.
        embedding_model (str): The name of the embedding model to use for generating embeddings.

    Returns:
        tuple: A tuple containing the vector database and the embeddings model.
    """

    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Extract text content from split documents
    document_texts = [doc.page_content for doc in split_documents]
    
    # Generate embeddings for the document texts
    document_embeddings = embeddings.embed_documents(document_texts)
    document_embeddings = np.array(document_embeddings, dtype=np.float32)
    
    # Normalize the embeddings
    document_embeddings = normalize(document_embeddings)

    # Ensure the embeddings are contiguous and of type float32
    document_embeddings = np.ascontiguousarray(document_embeddings, dtype=np.float32)

    # Create a FAISS index for the embeddings
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(document_embeddings)

    # Map document IDs to documents
    id_to_document = {str(i): doc for i, doc in enumerate(split_documents)}
    vector_database = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(id_to_document),
        index_to_docstore_id={i: str(i) for i in range(len(split_documents))},
    )

    # Save the vector database locally
    db_path = "vectorstore/db_faiss"
    os.makedirs("vectorstore", exist_ok=True)
    vector_database.save_local(db_path)

    print("- Documents inserted into FAISS vectorstore")
    return vector_database, embeddings