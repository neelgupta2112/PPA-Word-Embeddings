
import json
import gzip
from io import BytesIO
import boto3
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast
from collections import defaultdict
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from io import StringIO

keyword_db = pd.read_parquet("../Data/ppa_corpus_2025-02-03_1308/keywords_and_top_1000_edited.parquet")
forms = keyword_db[~keyword_db['page_text'].isna()]['poetic_form'].unique()
del keyword_db

TARGET_FORMS = set(forms)

EMBEDDING_DTYPE = np.float32

SELECTED_PARQUET = "selected_forms.parquet"
SELECTED_CSV = "selected_forms.csv"
OTHER_CSV = "other_forms_avg.csv"



S3_BUCKET = "ppa-embeddings-bucket"
S3_INPUT_PREFIX = "target_embeddings_expanded/"       
s3 = boto3.client(
    "s3",
    aws_access_key_id="AKIAYQNJSRWCQL32G6XW",
    aws_secret_access_key="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",
    region_name="us-west-2"
)



def stream_jsonl_gz_from_s3(s3_client, bucket, key):
    """
    Stream a .jsonl.gz file from S3 and yield dicts.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    with gzip.GzipFile(fileobj=obj["Body"], mode="r") as gz:
        for line in gz:
            yield json.loads(line)



all_other_records = defaultdict(lambda: {
    "sum": None,
    "count": 0
})

def update_other(form, embedding):
    emb = np.asarray(embedding, dtype=EMBEDDING_DTYPE)
    if all_other_records[form]["sum"] is None:
        all_other_records[form]["sum"] = emb.copy()
    else:
        all_other_records[form]["sum"] += emb
    all_other_records[form]["count"] += 1


selected_writer = None

def write_selected(row):
    global selected_writer
    table = pa.Table.from_pylist([row])

    if selected_writer is None:
        selected_writer = pq.ParquetWriter(
            SELECTED_PARQUET,
            table.schema,
            compression="zstd"
        )

    selected_writer.write_table(table)



paginator = s3.get_paginator("list_objects_v2")

for page in paginator.paginate(
    Bucket=S3_BUCKET,
    Prefix=S3_INPUT_PREFIX
):
    for obj in tqdm(
        page.get("Contents", []),
        desc="Reading S3 JSONL.GZ files"
    ):
        key = obj["Key"]
        if not key.endswith(".jsonl.gz"):
            continue

        for record in stream_jsonl_gz_from_s3(s3, S3_BUCKET, key):
            form = record.get("poetic_form")
            embedding = record.get("embedding")

            if form is None or embedding is None:
                continue  # skip malformed rows

            if form in TARGET_FORMS:
                write_selected(record)
            else:
                update_other(form, embedding)



if selected_writer is not None:
    selected_writer.close()


other_rows = []

for form, stats in all_other_records.items():
    if stats["count"] == 0:
        continue

    avg_embedding = (stats["sum"] / stats["count"]).tolist()

    other_rows.append({
        "poetic_form": form,
        "embedding": avg_embedding
    })


df_selected = pd.read_parquet(SELECTED_PARQUET)

other_rows.to_csv("most_common_words.csv", index = False)
print("Wrote most_common_words.csv")
# ---------- WRITE TO CSV IN MEMORY ----------
csv_buffer = StringIO()
df_selected.to_csv(csv_buffer, index=False)
s3.put_object(
    Bucket=S3_BUCKET,
    Key="processed/selected_forms.csv",  # S3 path
    Body=csv_buffer.getvalue()
)

csv_buffer = StringIO()
pd.DataFrame(other_rows).to_csv(csv_buffer, index=False)

s3.put_object(
    Bucket=S3_BUCKET,
    Key="processed/other_forms_avg.csv",  
    Body=csv_buffer.getvalue()
)

print(f"Selected forms written to: {SELECTED_CSV}")
print(f"Other forms written to: {OTHER_CSV}")
