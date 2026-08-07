import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from mcp_client import build_context_with_mcp, fetch_mcp_context

SYSTEM_PROMPT = """
You are a custom non-hallucinating AI assistant.
Answer the questions given based ONLY on the provided context.
If the answer is not in the context, say "Not within the provided context."
"""


def extract_keywords(text: str):
    """Extract simple keywords from a string by cleaning and filtering common stop words."""
    words = re.findall(r"\w+", text.lower())
    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "is", "are",
        "for", "on", "with", "from", "this", "that", "what", "why",
        "how", "when", "who", "do", "does", "can", "be", "my", "our"
    }

    return [word for word in words if word not in stop_words and len(word) > 2]


def detect_intent(question: str) -> str:
    """Map a question to a simple intent label for this workplace-guidelines app."""
    q = question.lower()

    if any(word in q for word in ["leave", "vacation", "time off", "absence"]):
        return "leave_request"
    if any(word in q for word in ["overtime", "hours", "working time", "schedule"]):
        return "overtime_policy"
    if any(word in q for word in ["contact", "support", "help", "who"]):
        return "support_contact"
    if any(word in q for word in ["discipline", "warning", "misconduct", "policy breach"]):
        return "disciplinary_policy"
    if any(word in q for word in ["attendance", "late", "absent", "punctuality"]):
        return "attendance_policy"

    return "general"


def rerank_documents(question: str, docs):
    """Rerank documents using both intent and keyword overlap."""
    if not docs:
        return docs

    intent = detect_intent(question)
    question_keywords = set(extract_keywords(question))
    if not question_keywords:
        return docs

    scored_docs = []
    for doc in docs:
        content = doc.page_content.lower()
        doc_keywords = set(extract_keywords(doc.page_content))
        overlap = len(question_keywords & doc_keywords)
        score = overlap / max(1, len(question_keywords))

        if intent in content:
            score += 0.5

        if any(keyword in content for keyword in question_keywords):
            score += 0.2

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs]


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
            search_kwargs={"k": 6}
        )

        docs = retriever.invoke(question)
        docs = rerank_documents(question, docs)
        docs = docs[:3]

        context = "\n\n".join(doc.page_content for doc in docs)
        mcp_context = fetch_mcp_context(question)
        context = build_context_with_mcp(context, mcp_context)

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        return response.content

    return rag_answer