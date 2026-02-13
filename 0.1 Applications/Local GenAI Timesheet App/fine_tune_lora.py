import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def main():
    # Check GPU memory
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load model with proper quantization config
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fixed quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16  # Fixed: use 'dtype' instead of 'torch_dtype'
    )

    print(f"Using device: cuda:0")
    print(f"Model loaded: {model_name} with 4-bit quantization")

    # Load dataset directly from JSONL
    dataset = load_dataset("json", data_files="synthetic_timesheet_data.jsonl")
    dataset = dataset["train"]

    print(f"Loaded {len(dataset)} samples from timesheet_data.jsonl")

    # Preprocessing function
    def preprocess_function(examples):
        """Preprocess the dataset for training"""
        inputs = ["Classify and estimate effort for this task: " + t for t in examples["text"]]
        targets = [
            f'{{"category": "{l}", "effort_hours": "{e}", "priority": "medium"}}'
            for l, e in zip(examples["label"], examples["effort"])
        ]
        
        model_inputs = tokenizer(
            inputs,
            max_length=256,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split dataset
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)["train"]
    print(f"Training samples: {len(train_dataset)}")

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v", "k", "o", "wi", "wo"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # FIXED: Windows-compatible training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./timesheet_model_optimized",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=3e-4,
        warmup_steps=50,
        logging_dir="./logs",
        logging_steps=2,
        save_strategy="epoch",
        bf16=True,
        dataloader_num_workers=0,  # FIXED: Set to 0 for Windows compatibility
        group_by_length=True,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        predict_with_generate=True,
        generation_max_length=128,
        dataloader_pin_memory=False,  # Better for Windows
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting optimized training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained("./timesheet_model_optimized")
    tokenizer.save_pretrained("./timesheet_model_optimized")
    print("✅ Optimized model saved to ./timesheet_model_optimized")

# CRITICAL: Windows multiprocessing protection
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows
    main()
