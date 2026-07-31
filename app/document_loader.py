from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_docs(path: str):
    loader = TextLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)