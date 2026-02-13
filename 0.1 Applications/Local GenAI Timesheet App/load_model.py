from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

# Model choice: lightweight for RTX 3050 Ti
model_name = "google/flan-t5-small"  

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# Test if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Model loaded: {model_name} with 8-bit quantization")
