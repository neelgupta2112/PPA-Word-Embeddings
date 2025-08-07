import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from itertools import islice
import json
import os
import gzip

def extract_usage_representations(text, tokenizer, model, device="cpu"):
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True) ### tokenizing the input
    input_ids = encoded["input_ids"].to(device) ## moving stuff to the gpu
    attention_mask = encoded["attention_mask"].to(device) ## moving stuff to the gpu
    # offset_mapping = encoded["offset_mapping"][0]

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) ## running the text through he model
        hidden_states = output.hidden_states  # Tuple: (layer, batch, seq_len, hidden_size)

    all_layers = torch.stack(hidden_states, dim=0)  # Shape: (layers, batch, seq_len, hidden) ## grabbing the layers 
    summed = all_layers.sum(dim=0)[0]  # sum across layers → (seq_len, hidden_size) ## sum all hidden layers

    # Map tokens to words (simple tokenizer-based match)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    usage_vectors = []

    for i, token in enumerate(tokens):
        word = token.lstrip("Ġ")  # Remove special prefix if tokenizer is Roberta-like
        usage_vectors.append((word.lower(), summed[i].cpu()))

    return usage_vectors


def is_semantically_meaningful(token):
    token = token.lower()
    STOPWORDS = {
    "the", "and", "for", "but", "with", "that", "this", "from", "not",
    "you", "are", "was", "were", "have", "has", "had", "she", "he", "they",
    "his", "her", "its", "our", "their", "will", "would", "can", "could"
    }
    return (
        token.isalpha() and
        len(token) > 2 and
        token not in STOPWORDS
    )

def page_iter(pages_file):
   with gzip.open(pages_file, 'rt', encoding='utf-8') as fh:
       for line in fh:
           yield json.loads(line)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

TARGET_COLLECTIONS = {"Literary", "Linguistic"}

with open("Data/ppa_corpus_2025-02-03_1308/ppa_metadata.json") as f:
    metadata = json.load(f)

metadata_index = {
    entry["source_id"]: entry for entry in metadata
    if "collections" in entry and any(c in TARGET_COLLECTIONS for c in entry["collections"])
}



# Save output
BATCH_SIZE = 1000
MAX_PAGES = None  
OUTPUT_DIR = "output_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

page_generator = page_iter("Data/ppa_corpus_2025-02-03_1308/ppa_pages.jsonl.gz")
batch_idx = 0
total_processed = 0

while True:
    batch = list(islice(page_generator, BATCH_SIZE))
    if not batch:
        break

    output_path = f"{OUTPUT_DIR}/usage_batch_{batch_idx:05}.jsonl.gz"
    if os.path.exists(output_path):
        print(f"Skipping batch {batch_idx}, already exists.")
        batch_idx += 1
        total_processed += len(batch)
        continue

    output_batch = []

    for example in tqdm(batch, desc=f"Batch {batch_idx}"):
        text = example.get("text")
        pid = example.get("id")
        wid = example.get("work_id")

        if not text or wid not in metadata_index:
            continue

        meta = metadata_index[wid]
        pub_year = meta.get("pub_year")
        collections = meta.get("collections")

        for word, usage_vector in extract_usage_representations(text, tokenizer, model, device=DEVICE):
            if not is_semantically_meaningful(word):
                continue
            output_batch.append({
                "word": word,
                "usage_vector": usage_vector.tolist(),
                "text": text,
                "id": pid,
                "work_id": wid,
                })

    output_path = f"{OUTPUT_DIR}/usage_batch_{batch_idx:05}.jsonl.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as f_out:
        for item in output_batch:
            f_out.write(json.dumps(item) + "\n")

    batch_idx += 1
    total_processed += len(batch)
    if MAX_PAGES and total_processed >= MAX_PAGES:
        break

