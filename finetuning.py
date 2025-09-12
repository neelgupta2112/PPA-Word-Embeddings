import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tqdm import tqdm
from itertools import islice
import json
import os
import gzip
import string
from datasets import Dataset

def page_iter(pages_file):
   with gzip.open(pages_file, 'rt', encoding='utf-8') as fh:
       for line in fh:
           yield json.loads(line)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(DEVICE)


TARGET_COLLECTIONS = {"Literary"}
EXCLUSION_COLLECTIONS = {"Dictionary", "Word Lists", "Typographically Unique"}


with open("Data/ppa_corpus_2025-02-03_1308/ppa_metadata.json") as f:
    metadata = json.load(f)


metadata_index = {
    entry["work_id"]: entry
    for entry in metadata
    if "collections" in entry
       and any(c in TARGET_COLLECTIONS for c in entry["collections"])
       and not any(c in EXCLUSION_COLLECTIONS for c in entry["collections"])
}



print(len(metadata_index))



# total_tokens = 0
# total_chunks = 0
# block_size = 512

# for page in tqdm(page_iter("Data/ppa_corpus_2025-02-03_1308/ppa_pages.jsonl.gz"), desc="Counting tokens"):
#     wid = page.get("work_id")
#     if wid not in metadata_index:
#         continue

#     text = page.get("text", "").strip()
#     if not text:
#         continue

#     tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
#     n_tokens = len(tokens)
#     total_tokens += n_tokens
#     total_chunks += n_tokens // block_size  # full blocks only

# print(f"Total tokens: {total_tokens:,}")
# print(f"Total 512-token chunks: {total_chunks:,}")
# print(f"Avg tokens per eligible source: {total_tokens / len(metadata_index):.0f}")




def pages_generator(file, allowed_ids):
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            page = json.loads(line)
            wid = page.get("work_id")
            text = page.get("text", "").strip()
            if wid in allowed_ids and text:
                yield {"text": text}

dataset = Dataset.from_generator(lambda: pages_generator("Data/ppa_corpus_2025-02-03_1308/ppa_pages.jsonl.gz", metadata_index.keys()))

### PARAM
block_size = 512

def tokenize_and_chunk(examples):
    tokenized = tokenizer(examples["text"], return_attention_mask=False, return_token_type_ids=False)
    input_ids = tokenized["input_ids"]
    chunks = [input_ids[i:i+block_size] for i in range(0, len(input_ids), block_size) if len(input_ids[i:i+block_size]) == block_size]
    return {"input_ids": chunks}

tokenized_dataset = dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.flatten() 

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) #PARAM


training_args = TrainingArguments(
    output_dir="./modernbert-literary-mlm",
    per_device_train_batch_size=16, ##PARAM
    num_train_epochs=3,  ## PARAM
    learning_rate=5e-5, ## PARAM
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100, ## PARAM
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)


trainer.train()


trainer.save_model("./finetuned_modernbert-literary-mlm")
tokenizer.save_pretrained("./fintuned_modernbert-literary-mlm")