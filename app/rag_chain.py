from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a custom non-hallucinating AI assistant.
Answer the questions given based ONLY on the provided context.
If the answer is not in the context, say "Not within the provided context."
"""


def create_rag_chain(vector_db):

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        think=False,              # Disable reasoning mode
        num_ctx=8192              # Optional: increase context window
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ]
    )

    def rag_answer(question: str):
        retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

        docs = retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        return response.content

    return rag_answer