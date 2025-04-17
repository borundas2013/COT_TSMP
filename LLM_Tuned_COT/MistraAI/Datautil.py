from datasets import load_dataset
import json
import os
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "deepseek-ai/deepseek-llm-7b-instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    qlora = True,
)


train_data_path = "./MistraAI/Data/newformat/training_small_combined_conversational_unsloth.jsonl"
valid_data_path = "./MistraAI/Data/newformat/validation_small_combined_conversational_unsloth.jsonl"

# Load the datasets with data_files parameter
train_dataset = load_dataset('json', data_files=train_data_path)
valid_dataset = load_dataset('json', data_files=valid_data_path)

print(train_dataset)
print(valid_dataset)

def format_chat(example):
    prompt = tokenizer.apply_chat_template(example["conversations"], tokenize=False)
    return {"prompt": prompt}

train_dataset = train_dataset.map(format_chat)
eval_dataset = valid_dataset.map(format_chat)

def tokenize(example):
    tokens = tokenizer(example["prompt"], padding="max_length", truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(tokenize, remove_columns=["conversations", "prompt"])
eval_dataset = eval_dataset.map(tokenize, remove_columns=["conversations", "prompt"])
print(train_dataset)
print(eval_dataset)








