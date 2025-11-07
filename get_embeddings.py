import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from itertools import islice
import json, os, gzip, string, boto3
from io import BytesIO
import gc
from threading import Thread, Lock
import time

# ------------------ CONFIG ------------------
BATCH_SIZE = 200
MAX_PAGES = None
OUTPUT_DIR = "output_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./finetuned_modernbert-literary-mlm"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
model = torch.compile(model)

S3_BUCKET = "ppa-embeddings-bucket"
S3_PREFIX = "embeddings_final/"
s3 = boto3.client("s3", region_name="us-west-2")

STOPWORDS = {
   "the", "and", "for", "but", "with", "that", "this", "from", "not",
  "you", "are", "was", "were", "have", "has", "had", "she", "he", "they",
   "his", "her", "its", "our", "their", "will", "would", "can", "could"
}

TARGET_COLLECTIONS = {"Literary", "Linguistic"}
EXCLUSION_COLLECTIONS = {"Dictionary", "Word Lists", "Original Bibliography"}

PAGES_FILE = "Data/ppa_corpus_2025-02-03_1308/ppa_pages.jsonl.gz"
METADATA_FILE = "Data/ppa_corpus_2025-02-03_1308/ppa_metadata.json"

# ------------------ LOAD METADATA ------------------
with open(METADATA_FILE) as f:
    metadata = json.load(f)

metadata_index = {
    entry["work_id"]: entry
    for entry in metadata
    if "collections" in entry
    and any(c in TARGET_COLLECTIONS for c in entry["collections"])
    and not any(c in EXCLUSION_COLLECTIONS for c in entry["collections"])
}

print(f"✅ Metadata filtered: {len(metadata_index)} works")

# ------------------ UTILS ------------------
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

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states

    summed = sum(hidden_states[-4:])[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special_tokens = set(tokenizer.all_special_tokens)

    usage_vectors = []
    current_word = ""
    current_vecs = []
    current_start = None
    current_end = None

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

def page_iter_filtered(pages_file, metadata_index):
    with gzip.open(pages_file, 'rt', encoding='utf-8') as fh:
        for line in fh:
            page = json.loads(line)
            if page.get("work_id") in metadata_index:
                yield page

# ------------------ ETA TRACKING ------------------
start_time = time.time()
lock = Lock()
completed_batches = 0
total_batches = None  # will be computed dynamically

def upload_to_s3(buffer, key):
    global completed_batches
    try:
        s3.upload_fileobj(buffer, S3_BUCKET, key)
        print(f"✅ Uploaded {key}")
    except Exception as e:
        print(f"❌ Upload failed for {key}: {e}")
    finally:
        with lock:
            completed_batches += 1
            if total_batches:
                elapsed = time.time() - start_time
                remaining = total_batches - completed_batches
                eta_min = elapsed / completed_batches * remaining / 60
                print(f"⏱ ETA remaining: {eta_min:.2f} min")

# ------------------ MAIN LOOP ------------------
page_generator = page_iter_filtered(PAGES_FILE, metadata_index)

# Estimate total batches
if not MAX_PAGES:
    total_pages = sum(1 for _ in page_iter_filtered(PAGES_FILE, metadata_index))
else:
    total_pages = min(MAX_PAGES, sum(1 for _ in page_iter_filtered(PAGES_FILE, metadata_index)))
total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Total batches: {total_batches}")

batch_idx = 0
total_processed = 0

while True:
    batch = list(islice(page_generator, BATCH_SIZE))
    if not batch:
        break

    s3_key = f"{S3_PREFIX}usage_batch_{batch_idx:05}.jsonl.gz"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print(f"Skipping batch {batch_idx}, already exists on S3.")
        batch_idx += 1
        total_processed += len(batch)
        with lock:
            completed_batches += 1
        continue
    except s3.exceptions.ClientError:
        pass

    batch = [ex for ex in batch if ex.get("text")]
    output_batch = []

    for example in tqdm(batch, desc=f"Batch {batch_idx}"):
        text = example.get("text")
        pid = example.get("id")
        wid = example.get("work_id")
        meta = metadata_index[wid]

        with torch.cuda.amp.autocast():
            usage_vecs = extract_usage_representations(text, tokenizer, model, device=DEVICE)

        output_batch.extend([
            {
                "word": w["word"],
                "usage_vector": w["vector"].tolist(),
                "char_start": w["char_start"],
                "char_end": w["char_end"],
                "id": pid,
                "work_id": wid,
            }
            for w in usage_vecs
        ])

    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="w") as gz_out:
        for item in output_batch:
            gz_out.write((json.dumps(item) + "\n").encode("utf-8"))
    buffer.seek(0)

    # Upload in background
    Thread(target=upload_to_s3, args=(buffer, s3_key)).start()

    del output_batch
    if batch_idx % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()

    batch_idx += 1
    total_processed += len(batch)

print("✅ All embeddings captured")
