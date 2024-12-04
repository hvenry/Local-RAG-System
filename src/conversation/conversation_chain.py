from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

def create_conversation_chain(retriever, llm_model="llama3.2", service_url="http://localhost:11434"):
    """Setup an interface using the specified LLM that is running locally and use prompts to make model answer using context"""

    # Initialize LLM
    llm = OllamaLLM(model=llm_model, base_url=service_url)
    
    # System prompt for contextualizing questions
    contextualize_question_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents."
    )
    contextualize_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_question_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_question_prompt
    )
    
    # System prompt for answering questions
    question_answer_system_prompt = (
        "You are a helpful assistant. Use only the provided context to answer the user's question. "
        "If the answer is not in the context, say 'I'm sorry, but I don't have that information.'"
    )
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Context: {context}\nQuestion: {input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(
        llm, question_answer_prompt, document_variable_name="context"
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Store for chat message history
    chat_history_store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in chat_history_store:
            chat_history_store[session_id] = ChatMessageHistory()
        return chat_history_store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("- Conversational chain created")
    return conversational_rag_chain