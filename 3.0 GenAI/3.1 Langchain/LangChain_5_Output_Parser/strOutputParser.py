from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(hf_token)

llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

# 1st report --> details Prompt
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables= ['topic']
)

#2nd prompt --> summry
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n  {text}',
    input_variables=['text']
)

prompt1 = template1.format(topic="black Hole")
result = model.invoke(prompt1)

prompt2 = template2.format(text=result.content)
result1 = model.invoke(prompt2)
print(result1.content)
