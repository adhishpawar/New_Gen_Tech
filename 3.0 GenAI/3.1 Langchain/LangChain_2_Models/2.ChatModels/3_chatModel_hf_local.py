from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

quant_config = BitsAndBytesConfig(load_in_8bit=True)


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

print("Model loaded, running inference...")
result = chat_model.invoke("What is the capital of India")
print("Inference done!")
print("Hi5")
print(result)          # Shows whole AIMessage
print(result.content)  # Shows just the text
print("Hi")
