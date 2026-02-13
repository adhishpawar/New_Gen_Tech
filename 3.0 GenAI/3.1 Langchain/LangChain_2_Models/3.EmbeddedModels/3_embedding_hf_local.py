import os
from langchain_huggingface import HuggingFaceEmbeddings

# Set local folder for HuggingFace model cache
os.environ["TRANSFORMERS_CACHE"] = r"E:\Personal Things\02 Projects\13 DeepLearning\LangChain_Models\3.EmbeddedModels"

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the captical of India"

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of west Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

vector = embedding.embed_query(text)

print(str(result))