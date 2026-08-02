from document_loader import load_and_split_docs
from vector_store import create_vector_store
from rag_chain import create_rag_chain


def main():
    print("� Loading documents...")
    chunks = load_and_split_docs("data/workers_guidelines.txt")

    print("🗂️ Creating vector store...")
    vector_db = create_vector_store(chunks)

    print("⚙️ Initializing RAG...")
    rag = create_rag_chain(vector_db)

    print("\n🚀 RAG is ready! Type 'exit' to quit.\n")

    while True:
        question = input("💬 Question: ")
        if question.lower() == "exit":
            break

        answer = rag(question)
        print(f"\n� Answer: {answer}\n")


if __name__ == "__main__":
    main()