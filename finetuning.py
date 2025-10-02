import torch
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tqdm import tqdm
import json
import gzip
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

print(f"Eligible works: {len(metadata_index)}")

def pages_generator(file, allowed_ids):
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            page = json.loads(line)
            wid = page.get("work_id")
            text = page.get("text", "").strip()
            if wid in allowed_ids and text:
                yield {"text": text}

dataset = Dataset.from_generator(lambda: pages_generator(
    "Data/ppa_corpus_2025-02-03_1308/ppa_pages.jsonl.gz",
    metadata_index.keys())
)

### PARAM
block_size = 512

def tokenize_and_chunk(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=block_size,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

tokenized_dataset = dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.flatten()

# Split into train/validation
dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=25)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./modernbert-literary-mlm",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    learning_rate=2.5e-5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

### Evaluate baseline
baseline_metrics = trainer.evaluate()
baseline_perplexity = math.exp(baseline_metrics["eval_loss"])
print(f"\nBaseline perplexity: {baseline_perplexity:.2f}\n")

### Train
trainer.train()

### Evaluate finetuned model
finetuned_metrics = trainer.evaluate()
finetuned_perplexity = math.exp(finetuned_metrics["eval_loss"])
print(f"\nFinetuned perplexity: {finetuned_perplexity:.2f}\n")

### Save model + tokenizer
trainer.save_model("./finetuned_modernbert-literary-mlm")
tokenizer.save_pretrained("./finetuned_modernbert-literary-mlm")
