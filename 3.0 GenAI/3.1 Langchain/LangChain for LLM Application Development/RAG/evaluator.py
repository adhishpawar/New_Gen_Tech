# Evaluation logic (LLM-as-a-judge)

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

evaluator_llm = ChatOpenAI(
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    temperature=0
)

eval_prompt = PromptTemplate(
    input_variables=["question", "answer", "reference"],
    template="""
            You are an evaluator. Your task is to grade the following AI-generated answer.

            Question: {question}
            Answer: {answer}
            Reference Answer: {reference}

            Grade the answer on:
            1. Correctness (0-10)
            2. Relevance (0-10)
            3. Completeness (0-10)


            Respond in JSON format:
            {{
            "correctness": score,
            "relevance": score,
            "completeness": score,
            "feedback": "short feedback here"
            }}
                
            """
)

def evaluate_answer(question, generated_answer, reference_answer):
    prompt_str = eval_prompt.format(
        question=question,
        answer=generated_answer,
        reference=reference_answer
    )

    result = evaluator_llm.invoke(prompt_str)
    return result.content