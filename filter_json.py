import json
import os
import gzip


target_words = []


input_dir = "output_batches"
output_dir = "cleaned_batches"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jsonl.gz"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with gzip.open(input_path, "rt", encoding="utf-8") as fin, \
         gzip.open(output_path, "wt", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)
            if data.get("word") in target_words:
                fout.write(json.dumps(data) + "\n")

    print(f"Filtered {filename} -> {output_path}")