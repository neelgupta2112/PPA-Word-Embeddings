import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import json, gzip, os, string, boto3

# ---------------- CONFIG ----------------
MODEL_DIR = "./finetuned_modernbert-literary-mlm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S3_BUCKET = "ppa-embeddings-bucket"
S3_PREFIX = "target_embeddings_expanded/"
LOCAL_TMP = "tmp_batches"
CONTEXT_CHARS = 100
BATCH_SIZE = 200

os.makedirs(LOCAL_TMP, exist_ok=True)
s3 = boto3.client("s3")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
model = torch.compile(model)

STOPWORDS = {
    "the", "and", "for", "but", "with", "that", "this", "from", "not",
    "you", "are", "was", "were", "have", "has", "had", "she", "he", "they",
    "his", "her", "its", "our", "their", "will", "would", "can", "could"
}

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Data/ppa_corpus_2025-02-03_1308/ppa_keyword_db.csv")
df = df[["page_id", "poetic_form", "spelling", "page_text", "work_id"]] # "title", "author", "pub_year"]]

# map form -> list of spellings
form_to_spellings = {
    f: [s.lower().strip(string.punctuation) for s in spellings]
    for f, spellings in df.groupby("poetic_form")["spelling"].agg(list).items()
}

# ---------------- GROUP BY PAGE ----------------
pages = df.groupby("page_id", sort=False).agg({
    "page_text": "first",
    "spelling": lambda x: list(x),
    "work_id": "first",
    # "title": "first",
    # "author": "first",
    "pub_year": "first"
}).reset_index()

# targets now only contain spellings
pages["targets"] = pages["spelling"].tolist()
pages = pages.drop(columns=["spelling"])
print(f"✅ Unique pages to embed: {len(pages)}")

# ---------------- HELPER ----------------
def extract_usage_representations(text, tokenizer, model, device="cpu", skip_stopwords=True):
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = encoded["offset_mapping"][0]

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states

    summed = sum(hidden_states[-4:])[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special_tokens = set(tokenizer.all_special_tokens)

    usage_vectors = []
    current_word, current_vecs = "", []
    current_start, current_end = None, None

    for i, token in enumerate(tokens):
        if token in special_tokens:
            continue
        is_new_word = token.startswith("Ġ") or i == 0 or token.startswith("Ċ")
        if is_new_word and current_word:
            word_vec = torch.stack(current_vecs).mean(dim=0)
            clean_word = current_word.lower().strip(string.punctuation)
            if clean_word and (not skip_stopwords or clean_word not in STOPWORDS):
                usage_vectors.append({
                    "word": clean_word,
                    "vector": word_vec.detach().cpu().numpy(),
                    "char_start": current_start,
                    "char_end": current_end
                })
            current_vecs = []
        if not current_vecs:  # first real subtoken of the current word
            current_start = offsets[i][0].item()

        if is_new_word:
            current_word = token.lstrip("ĠĊ")
            current_start = offsets[i][0].item()
            current_end = offsets[i][1].item()
            current_vecs.append(summed[i])
        else:
            current_word += token
            current_end = offsets[i][1].item()
            current_vecs.append(summed[i])

    if current_word:
        word_vec = torch.stack(current_vecs).mean(dim=0)
        clean_word = current_word.lower().strip(string.punctuation)
        if clean_word and (not skip_stopwords or clean_word not in STOPWORDS):
            usage_vectors.append({
                "word": clean_word,
                "vector": word_vec.detach().cpu().numpy(),
                "char_start": current_start,
                "char_end": current_end
            })
    return usage_vectors

def s3_object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

# ---------------- MAIN LOOP ----------------
num_batches = (len(pages) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(num_batches):
    s3_key = f"{S3_PREFIX}batch_{batch_idx:05d}.jsonl.gz"
    if s3_object_exists(S3_BUCKET, s3_key):
        print(f"🟡 Skipping existing batch {batch_idx}")
        continue

    start = batch_idx * BATCH_SIZE
    end = min((batch_idx + 1) * BATCH_SIZE, len(pages))
    batch_pages = pages.iloc[start:end]
    local_path = os.path.join(LOCAL_TMP, f"batch_{batch_idx:05d}.jsonl.gz")

    with gzip.open(local_path, "wt") as f_out:
        for _, row in tqdm(batch_pages.iterrows(), total=len(batch_pages), desc=f"Batch {batch_idx}"):
            text = row["page_text"]
            target_spellings = {s.lower().strip(string.punctuation) for s in row["targets"]}
            page_id = row["page_id"]

            all_usages = extract_usage_representations(text, tokenizer, model, device=DEVICE)

            for u in all_usages:
                if u["word"] in target_spellings:
                    start = u["char_start"]
                    end = u["char_end"]

                    left = max(0, start - CONTEXT_CHARS)
                    right = min(len(text), end + CONTEXT_CHARS)
                    context_snippet = text[left:right]

                    # map spelling back to a form (pick the first form that contains this spelling)
                    form_candidates = [f for f, spellings in form_to_spellings.items() if u["word"] in [s.lower() for s in spellings]]
                    form_value = form_candidates[0] if form_candidates else None

                    json_obj = {
                        "page_id": page_id,
                        "work_id": row["work_id"],
                        # "title": row["title"],
                        # "author": row["author"],
                        # "pub_year": row["pub_year"],
                        "poetic_form": form_value,
                        "spelling": u["word"],
                        "char_start": start,
                        "char_end": end,
                        "context": context_snippet,
                        "embedding": u["vector"].tolist()
                    }
                    f_out.write(json.dumps(json_obj) + "\n")

    s3.upload_file(local_path, S3_BUCKET, s3_key)
    os.remove(local_path)
    print(f"✅ Uploaded {s3_key}")

print("🎉 All batches complete and uploaded to S3.")
