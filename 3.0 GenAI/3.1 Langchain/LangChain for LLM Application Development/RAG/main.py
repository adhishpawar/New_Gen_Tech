from rag_pipeline import build_rag_pipeline
from evaluator import evaluate_answer


file_path = r"E:\02 Projects\13 DeepLearning\LangChain for LLM Application Development\Basic RAG Code\demo.txt"


# Build the RAG pipeline
qa_chain = build_rag_pipeline(file_path)

##Question
question = "What is the main Topic?"

##Result
result = qa_chain.invoke({"query": question})
generated_answer = result['result']

##Now Evalution --> passing reference answer  
reference_answer = "The main topic of the document is about Prime Minister of India"

##Running evalution
evaluation = evaluate_answer(question, generated_answer, reference_answer)

# Display results
print("\n🔹 Question:", question)
print("✅ Generated Answer:", generated_answer)
print("📊 Evaluation Result:", evaluation)