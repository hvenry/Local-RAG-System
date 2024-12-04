import os
from typing import List
from langchain.schema import Document
from processing.data_processing import prepare_and_split_documents
from processing.vectorstore import load_documents_into_vectorstore
from conversation.conversation_chain import create_conversation_chain
from utils import calculate_similarity_score

def main() -> None:
    # Directory where documents are stored (default is /data)
    data_directory = "data"
    os.makedirs(data_directory, exist_ok=True)

    print("\nStep 1: Preparing and splitting documents...")
    split_documents: List[Document] = prepare_and_split_documents(data_directory)

    print("\nStep 2: Loading documents into the vector database...")
    vector_database, embeddings = load_documents_into_vectorstore(split_documents)

    print("\nStep 3: Setting up the conversational RAG system...")
    retriever = vector_database.as_retriever()
    retriever.search_type = "similarity"
    retriever.search_kwargs["k"] = 3 
    conversational_rag_chain = create_conversation_chain(retriever)

    print("\n\tRAG system setup complete!\n")

    # Conversation Loop 
    while True:
        user_question: str = input("RAG System: Ask me a question about my documents! (type 'exit' to quit).\n\n>>> ")

        terminating_strings = ["exit", "quit", "bye"]

        if user_question.lower() in terminating_strings:
            break

        # Retrieve documents and their similarity scores
        documents_and_scores = vector_database.similarity_search_with_score(user_question, k=3)

        print("\n\tTop Similarity Retrieved Documents:\n")

        for i, (document, score) in enumerate(documents_and_scores):
            print(f"\tDocument {i + 1} Score: {score}")
            print(f"\tDocument snippet:\n--------------------------------------------------\n{document.page_content[:45]}...\n--------------------------------------------------\n")

        # Generate response using the conversational RAG chain
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": "user_session"}},
        )
        answer = response["answer"]

        # Calculate similarity score between the answer and context documents
        context_documents: List[Document] = [doc for doc, _ in documents_and_scores]
        max_similarity_score: float = calculate_similarity_score(answer, context_documents, embeddings)
        print(f"\t< Max similarity between answer and context documents: {max_similarity_score} >\n")
        print(f"RAG System Response: {answer}\n")

if __name__ == "__main__":
    main()