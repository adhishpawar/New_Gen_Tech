from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prompts import QA_PROMPT

class SimpleAgent:
    """
    A tiny agent: fetch evidence via retriever, build context block, and ask LLM with a QA prompt.
    This shows "agent-style" tool orchestration without using LangChain's agent framework.
    """
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0.0, openai_api_key=None)  # pass key in rag pipeline normally

    def run(self, retriever, question: str, k: int = 4):
        docs = retriever.get_relevant_documents(question)[:k]
        context_parts = []
        for d in docs:
            page = d.metadata.get("page", "unknown")
            context_parts.append(f"[page {page}]\n{d.page_content}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context found."

        prompt = QA_PROMPT.format(context=context, question=question)
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(input_variables=["input"], template="{input}"))
        return chain.run(prompt)
