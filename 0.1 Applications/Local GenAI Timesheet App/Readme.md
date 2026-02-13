# 🤖 Local GenAI Timesheet Classifier

A fine-tuned T5-small model running locally on RTX 3050 Ti for automated timesheet task categorization and effort estimation.

## 🚀 Features

- Local LLM: Runs entirely on your hardware (no API calls)
- Memory Efficient: 4-bit quantization + LoRA fine-tuning
- Fast Training: ~45 seconds for 5 epochs on 10k samples
- Real-time Inference: <0.5s per task classification
- Production Ready: Django integration planned

## 🛠️ Tech Stack

| Component   | Technology                        |
|-------------|-----------------------------------|
| Model       | google/flan-t5-small (77M params) |
| Fine-tuning | LoRA (Parameter-Efficient)        |
| Quantization| 4-bit, bitsandbytes               |
| Framework   | HuggingFace Transformers + PEFT   |
| GPU         | NVIDIA RTX 3050 Ti (4GB VRAM)     |
| OS          | Windows 11 + CUDA                 |

## 📋 Requirements

### Hardware
- NVIDIA GPU with 4GB+ VRAM
- 8GB+ System RAM
- CUDA-compatible GPU

### Software
- Python 3.8+
- CUDA Toolkit 11.8+
- Pip packages (see below)

## 🔧 Installation

### 1. Clone Repository
git clone <your-repo-url>
cd Local-GenAI-Timesheet-App

text

### 2. Create Virtual Environment
python -m venv ts
ts\Scripts\activate

Or: source ts/bin/activate
text

### 3. Install Dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes peft --upgrade

text

### 4. Verify GPU Setup
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

text

## 🏃‍♂️ Quick Start

### Prepare Training Data
- Use `synthetic_timesheet_data.jsonl` (10k+ timesheet rows).
- Or generate your own using `generate_data.py`.

### Train Model
python fine_tune_lora.py

text

### Test Inference
python test_model.py

text

## 🎯 Model Capabilities

- Classifies timesheet entries into categories (Bug Fix, Testing, etc.).
- Estimates effort from natural text description.
- Output is structured as JSON.

**Example Input:**
> Fixed authentication bug in Django REST API

**Example Output:**
{"category": "Bug Fix", "effort_hours": "2 hrs", "priority": "medium"}

text

## 🔧 Configuration

Key parameters (`fine_tune_lora.py`):

lora_config = LoraConfig(
r=16,
lora_alpha=32,
target_modules=["q", "v", "k", "o", "wi", "wo"],
lora_dropout=0.1,
)
training_args = Seq2SeqTrainingArguments(
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
num_train_epochs=5,
learning_rate=3e-4,
)

text

## 🎯 Production Integration

- Django REST API endpoint for timesheet classification
- Supports automated logging, effort estimation, and category assignment

## 🛠️ Troubleshooting

**CUDA Error**
- Check CUDA drivers and PyTorch installation

**Windows Multiprocessing Error**
- Use:
if name == 'main':
multiprocessing.freeze_support()
main()

text

**Out of Memory**
- Lower batch size or gradient accumulation steps

**Poor Output**
- Add more training samples
- Tune prompt format and training epochs

## 📈 Next Steps

- [ ] Add evaluation metrics (BLEU, ROUGE)
- [ ] Integrate with web dashboards
- [ ] Expand language and category support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and open PR

## 📄 License

MIT License

## 🙏 Acknowledgments

- HuggingFace
- Microsoft bitsandbytes
- Google T5
- Open source community

---

*Last updated: September 2025*