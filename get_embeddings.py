import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from itertools import islice
import json
import os
import gzip
import string


# def is_semantically_meaningful(token):
#     token = token.lower()
#     STOPWORDS = {
#     "the", "and", "for", "but", "with", "that", "this", "from", "not",
#     "you", "are", "was", "were", "have", "has", "had", "she", "he", "they",
#     "his", "her", "its", "our", "their", "will", "would", "can", "could"
#     }
#     return (
#         token.isalpha() and
#         len(token) > 2 and
#         token not in STOPWORDS
#     )

# def extract_usage_representations(text, tokenizer, model, device="cpu"):
#     encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True) ### tokenizing the input
#     input_ids = encoded["input_ids"].to(device) ## moving stuff to the gpu
#     attention_mask = encoded["attention_mask"].to(device) ## moving stuff to the gpu
#     offsets = encoded["offset_mapping"][0]

#     with torch.no_grad():
#         output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) ## running the text through he model
#         hidden_states = output.hidden_states  # Tuple: (layer, batch, seq_len, hidden_size)

#     all_layers = torch.stack(hidden_states, dim=0)  # Shape: (layers, batch, seq_len, hidden) ## grabbing the layers 
#     summed = all_layers.sum(dim=0)[0]  # sum across layers → (seq_len, hidden_size) ## sum all hidden layers
#     # summed = all_layers[-4:].sum(dim=0)[0] maybe SUM ONLY LAST 4 LAYERS?? A lot of papers do this for BERT but 4 seems random and model specific


#     # Map tokens to words (simple tokenizer-based match)
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     usage_vectors = []

#     for i, token in enumerate(tokens):
#         word = token.lstrip("Ġ")  # Remove special prefix if tokenizer is Roberta-like
#         char_start, char_end = offsets[i].tolist()
#         usage_vectors.append({
#             "word": word.lower(),
#             "vector": summed[i].cpu(),
#             "token_idx": i,
#             "char_start": char_start,
#             "char_end": char_end
#         })

#     return usage_vectors



def extract_usage_representations(text, tokenizer, model, device="cpu", skip_stopwords=True):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = encoded["offset_mapping"][0]

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = output.hidden_states  # (layers, batch, seq_len, hidden_size)

    all_layers = torch.stack(hidden_states, dim=0)  # (layers, batch, seq_len, hidden)
    summed = all_layers.sum(dim=0)[0]  # (seq_len, hidden_size)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special_tokens = set(tokenizer.all_special_tokens)

    usage_vectors = []
    current_word = ""
    current_vecs = []
    current_start = None
    current_end = None

    STOPWORDS = {
       "the", "and", "for", "but", "with", "that", "this", "from", "not",
      "you", "are", "was", "were", "have", "has", "had", "she", "he", "they",
       "his", "her", "its", "our", "their", "will", "would", "can", "could"
    }

    for i, token in enumerate(tokens):
        # skip special tokens
        if token in special_tokens:
            continue

        # detect new word
        is_new_word = token.startswith("Ġ") or i == 0

        if is_new_word and current_word:
            # aggregate previous word
            word_vec = torch.stack(current_vecs).mean(dim=0)

            # strip punctuation + lowercase
            clean_word = current_word.lower().strip(string.punctuation)

            if clean_word and (not skip_stopwords or clean_word not in STOPWORDS):
                usage_vectors.append({
                    "word": clean_word,
                    "vector": word_vec.cpu(),
                    "char_start": current_start,
                    "char_end": current_end
                })

            current_vecs = []

        if is_new_word:
            current_word = token.lstrip("Ġ")
            current_start = offsets[i][0].item()
            current_end = offsets[i][1].item()
            current_vecs.append(summed[i])
        else:
            current_word += token
            current_end = offsets[i][1].item()
            current_vecs.append(summed[i])

    # Handle last word
    if current_word:
        word_vec = torch.stack(current_vecs).mean(dim=0)
        clean_word = current_word.lower().strip(string.punctuation)
        if clean_word and (not skip_stopwords or clean_word not in STOPWORDS):
            usage_vectors.append({
                "word": clean_word,
                "vector": word_vec.cpu(),
                "char_start": current_start,
                "char_end": current_end
            })

    return usage_vectors


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

        for word_info in extract_usage_representations(text, tokenizer, model, device=DEVICE):
            word = word_info['word']
            output_batch.append({
                "word": word,
                "usage_vector": word_info["vector"].tolist(),
                "char_start": word_info["char_start"],
                "char_end": word_info["char_end"], 
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

