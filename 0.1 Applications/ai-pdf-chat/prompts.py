from langchain.prompts import PromptTemplate

# Prompt used to answer using retrieved context
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant that answers questions using ONLY the provided CONTEXT.
                If context doesn't contain the answer, say "I don't know" and be concise.
                Use the context to answer and provide short references (like page numbers) when possible.

                CONTEXT:
                {context}

                QUESTION:
                {question}

                Answer:
            """
)

# A prompt to condense follow-up question into standalone form (for conversational chains)
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
                Given the conversation history and a follow up question, rephrase the follow up question to be a standalone question.
                Chat history:
                {chat_history}

                Follow up question: {question}

                Standalone question:
            """
)
