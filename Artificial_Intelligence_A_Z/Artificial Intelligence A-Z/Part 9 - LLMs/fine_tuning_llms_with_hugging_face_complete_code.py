# Fine-Tuning LLMs with Hugging Face

# Step 1: Installing the required packages and importing the libraries

# Installing the required packages

# !pip uninstall accelerate peft bitsandbytes transformers trl -y
# !pip install accelerate peft==0.13.2 bitsandbytes transformers trl==0.12.0
# !pip install huggingface_hub

# Importing the libraries

import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

# Step 2: Loading the model

llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "aboonaji/llama2finetune-v2",
                                                   quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = getattr(torch, "float16"), bnb_4bit_quant_type = "nf4"))
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

# Step 3: Loading the tokenizer

llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "aboonaji/llama2finetune-v2", trust_remote_code = True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Step 4: Setting the training arguments

training_arguments = TrainingArguments(output_dir = "./results", per_device_train_batch_size = 4, max_steps = 100)

# Step 5: Creating the Supervised Fine-Tuning trainer

llama_sft_trainer = SFTTrainer(model = llama_model,
                               args = training_arguments,
                               train_dataset = load_dataset(path = "aboonaji/wiki_medical_terms_llam2_format", split = "train"),
                               tokenizer = llama_tokenizer,
                               peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 64, lora_alpha = 16, lora_dropout = 0.1),
                               dataset_text_field = "text")

# Step 6: Training the model

# Warning: To run the training you first need to sign up here: https://wandb.ai/authorize?ref=models
# And then you will directly find in the main page your API key which you will enter in the output below.
llama_sft_trainer.train()

# Step 7: Chatting with the model

user_prompt = "Please tell me about Bursitis"
text_generation_pipeline = pipeline(task = "text-generation", model = llama_model, tokenizer = llama_tokenizer, max_length = 500)
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])