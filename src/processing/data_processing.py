from typing import List
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def prepare_and_split_documents(directory: str) -> List[Document]:
    """
    Load and split documents from the specified directory.

    This function loads documents of various types (PDF, DOCX, CSV) from the given directory,
    and splits them into smaller chunks for easier processing and embedding.

    Args:
        directory (str): The directory where the documents are stored.

    Returns:
        List[Document]: A list of split document chunks.
    """

    # Loaders for different document types
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader),
        DirectoryLoader(directory, glob="**/*.docx", show_progress=True),
        DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader),
    ]

    documents: List[Document] = []
    for loader in loaders:
        # Load documents using the loader
        data = loader.load()
        documents.extend(data)

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256, separators=["\n\n", "\n", " "]
    )
    split_documents = splitter.split_documents(documents)
    print(f"- Documents have been split into {len(split_documents)} passages")
    return split_documents