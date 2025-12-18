# import json
# import os
# import gzip


# target_words = []


# input_dir = "output_batches"
# output_dir = "cleaned_batches"
# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(input_dir):
#     if not filename.endswith(".jsonl.gz"):
#         continue

#     input_path = os.path.join(input_dir, filename)
#     output_path = os.path.join(output_dir, filename)

#     with gzip.open(input_path, "rt", encoding="utf-8") as fin, \
#          gzip.open(output_path, "wt", encoding="utf-8") as fout:

#         for line in fin:
#             data = json.loads(line)
#             if data.get("word") in target_words:
#                 fout.write(json.dumps(data) + "\n")

#     print(f"Filtered {filename} -> {output_path}")







import json
import gzip
from io import BytesIO
import boto3

S3_BUCKET = "ppa-embeddings-bucket"
S3_INPUT_PREFIX = "embeddings/"        # where your original batches are
S3_OUTPUT_PREFIX = "filtered_embeddings/"  # where filtered batches will go

s3 = boto3.client("s3", region_name="us-west-2")

# list of target words
target_words = ["love", "war", "freedom"]  # example, fill in your actual list
target_set = set(target_words)  # for faster lookup

# List objects in S3 input prefix
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_INPUT_PREFIX)
for obj in response.get("Contents", []):
    key = obj["Key"]
    if not key.endswith(".jsonl.gz"):
        continue

    print(f"Processing {key}...")

    # Download the object into memory
    obj_bytes = BytesIO()
    s3.download_fileobj(S3_BUCKET, key, obj_bytes)
    obj_bytes.seek(0)

    # Prepare output buffer
    out_buffer = BytesIO()
    
    with gzip.GzipFile(fileobj=obj_bytes, mode="r") as fin, \
         gzip.GzipFile(fileobj=out_buffer, mode="w") as fout:

        for line in fin:
            data = json.loads(line.decode("utf-8"))
            if data.get("word") in target_set:
                fout.write((json.dumps(data) + "\n").encode("utf-8"))

    # Upload filtered batch to S3
    out_buffer.seek(0)
    output_key = key.replace(S3_INPUT_PREFIX, S3_OUTPUT_PREFIX)
    s3.upload_fileobj(out_buffer, S3_BUCKET, output_key)
    print(f"✅ Uploaded filtered batch to s3://{S3_BUCKET}/{output_key}")
