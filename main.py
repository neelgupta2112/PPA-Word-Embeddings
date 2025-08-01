import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from itertools import islice

def extract_usage_representations(text, target_words, tokenizer, model, device="cpu"):
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offset_mapping = encoded["offset_mapping"][0]

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states  # Tuple: (layer, batch, seq_len, hidden_size)

    all_layers = torch.stack(hidden_states, dim=0)  # Shape: (layers, batch, seq_len, hidden)
    summed = all_layers.sum(dim=0)[0]  # sum across layers → (seq_len, hidden_size)

    # Map tokens to words (simple tokenizer-based match)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    usage_vectors = []

    for i, token in enumerate(tokens):
        word = token.lstrip("Ġ")  # Remove special prefix if tokenizer is Roberta-like
        if word.lower() in target_words:
            usage_vectors.append((word.lower(), summed[i].cpu()))

    return usage_vectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "GroNLP/bert-base-modern"  # Replace with actual ModernBERT name

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

target_words = ['ballad', 'poem', 'era', 'sounds']
TARGET_COLLECTIONS = {"Literary", "Linguistic"}

with open("data/ppa_metadata.json") as f:
    metadata = json.load(f)

metadata_index = {
    entry["source_id"]: entry for entry in metadata
    if "collections" in entry and any(c in TARGET_COLLECTIONS for c in entry["collections"])
}

# Process corpus
output = []
with open("data/corpus.jsonl") as f:
    for line in tqdm(islice(f, 1000))
        example = json.loads(line)
        text = example.get("text")
        sid = example.get("source_id")

        if sid not in metadata_index:
            continue

        meta = metadata_index[sid]
        pub_year = meta.get("pub_year")
        collections = meta.get("collections")

        for word, usage_vector in extract_usage_representations(text, target_words, tokenizer, model, device=DEVICE):
            output.append({
                "word": word,
                "usage_vector": usage_vector.tolist(),
                "date": pub_year,
                "text": text
            })

# Save output
with open("output_usages.jsonl", "w") as out_f:
    for item in output:
        out_f.write(json.dumps(item) + "\n")