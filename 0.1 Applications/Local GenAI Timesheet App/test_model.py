import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def main():
    print("🚀 Loading your fine-tuned timesheet classifier...")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model = PeftModel.from_pretrained(base_model, "./timesheet_model_optimized")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    def classify_task(task):
        input_text = f"Classify and estimate effort for this task: {task}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Test cases
    tasks = [
        "Fixed authentication bug in Django REST API",
        "Implemented Redis caching for user sessions",
        "Created unit tests for payment processing module",
        "Deployed microservice to AWS ECS with monitoring"
    ]
    
    print("\n🎯 Testing Your AI Timesheet Classifier:")
    print("=" * 60)
    
    for i, task in enumerate(tasks, 1):
        result = classify_task(task)
        print(f"\n{i}. Task: {task}")
        print(f"   AI Classification: {result}")
        print("-" * 60)
    
    print("\n🎉 Your local GenAI timesheet classifier is working!")

if __name__ == '__main__':
    main()
