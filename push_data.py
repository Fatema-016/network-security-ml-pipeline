import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from dotenv import load_dotenv

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────
MONGO_DB_NAME    = "network_security"
MONGO_COLLECTION = "network_data"
DATA_DIR         = "Network_Data"
BATCH_SIZE       = 5000
CHUNK_SIZE       = 50000

# Sample 40% of each file — stratified to preserve attack ratios
SAMPLE_FRACTION  = 0.4

# Target files — chosen for maximum attack type diversity
TARGET_FILES = [
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

# Top 10 features from CICIDS 2017 research
# Source: Sharafaldin et al. 2018 — "Toward Generating a New Intrusion
# Detection Dataset and Intrusion Traffic Characterization"
SELECTED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd PSH Flags",
    "Bwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Packet Length Mean",
    "Label"
]


# ── MongoDB connection ─────────────────────────────────────────────
def get_mongo_client() -> MongoClient:
    try:
        username = quote_plus(os.getenv("MONGO_USERNAME"))
        password = quote_plus(os.getenv("MONGO_PASSWORD"))
        cluster  = os.getenv("MONGO_CLUSTER")
        app_name = os.getenv("MONGO_APP_NAME")

        uri = (
            f"mongodb+srv://{username}:{password}"
            f"@{cluster}/?appName={app_name}"
        )
        client = MongoClient(uri, server_api=ServerApi("1"))
        client.admin.command("ping")
        logger.info("Successfully connected to MongoDB Atlas.")
        return client
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ── Data cleaning ──────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    ETL cleaning steps:
    1. Strip whitespace from column names
    2. Remove duplicate columns
    3. Keep only top 10 selected features + Label
    4. Replace inf/-inf with NaN
    5. Drop fully empty rows
    6. Binary encode Label: BENIGN=0, attacks=1
    """
    try:
        # 1. Strip column name whitespace
        df.columns = df.columns.str.strip()

        # 2. Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # 3. Keep only selected features
        available = [c for c in SELECTED_FEATURES if c in df.columns]
        missing   = [c for c in SELECTED_FEATURES if c not in df.columns]
        if missing:
            logger.warning(f"Features not found in chunk: {missing}")
        df = df[available]

        # 4. Replace infinity values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 5. Drop fully empty rows
        df.dropna(how="all", inplace=True)

        # 6. Binary encode label
        df["Label"] = df["Label"].str.strip()
        df["Label"] = df["Label"].apply(
            lambda x: 0 if str(x).upper() == "BENIGN" else 1
        )

        return df
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ── Push one file ──────────────────────────────────────────────────
def push_file_to_mongodb(filepath: str, collection) -> int:
    """
    Read CSV in chunks, clean, stratified sample, push to MongoDB.
    Returns total records inserted.
    """
    try:
        filename   = os.path.basename(filepath)
        total_rows = sum(1 for _ in open(filepath)) - 1
        logger.info(f"Reading file : {filename}")
        logger.info(f"{filename}  total rows: {total_rows:,}")
        logger.info(
            f"{filename}  will sample {SAMPLE_FRACTION*100:.0f}% "
            f"≈ {int(total_rows * SAMPLE_FRACTION):,} rows"
        )

        inserted     = 0
        chunk_number = 0
        benign_count = 0
        attack_count = 0

        for chunk in pd.read_csv(
            filepath,
            low_memory=False,
            chunksize=CHUNK_SIZE
        ):
            chunk_number += 1
            chunk = clean_dataframe(chunk)

            if chunk.empty:
                logger.warning(
                    f"{filename}  chunk {chunk_number} empty, skipping."
                )
                continue

            # Stratified sample within each chunk
            # preserves BENIGN/ATTACK ratio
            benign  = chunk[chunk["Label"] == 0].sample(
                frac=SAMPLE_FRACTION, random_state=42
            )
            attacks = chunk[chunk["Label"] == 1].sample(
                frac=SAMPLE_FRACTION, random_state=42
            )
            chunk = pd.concat([benign, attacks]).sample(
                frac=1, random_state=42  # shuffle
            ).reset_index(drop=True)

            benign_count += len(benign)
            attack_count += len(attacks)

            if chunk.empty:
                continue

            records = chunk.to_dict(orient="records")

            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i : i + BATCH_SIZE]
                collection.insert_many(batch)
                inserted += len(batch)

            logger.info(
                f"{filename}  chunk {chunk_number} done | "
                f"inserted so far: {inserted:,}"
            )

        logger.info(
            f"{filename}  COMPLETE | {inserted:,} records "
            f"(BENIGN: {benign_count:,} | ATTACK: {attack_count:,})"
        )
        return inserted

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ── Main ───────────────────────────────────────────────────────────
def main():
    try:
        client     = get_mongo_client()
        db         = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION]

        # Drop existing collection — fresh insert every run
        collection.drop()
        logger.info(f"Dropped '{MONGO_COLLECTION}' — fresh insert.")

        logger.info(
            f"Target files    : {len(TARGET_FILES)}\n"
            + "\n".join(f"   {f}" for f in TARGET_FILES)
        )
        logger.info(f"Features stored : {len(SELECTED_FEATURES)-1} + Label")
        logger.info(f"Sample fraction : {SAMPLE_FRACTION*100:.0f}% stratified")

        total_inserted = 0
        for filename in TARGET_FILES:
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                logger.warning(f"File not found, skipping: {filepath}")
                continue
            count = push_file_to_mongodb(filepath, collection)
            total_inserted += count

        # Final label distribution
        benign  = collection.count_documents({"Label": 0})
        attacks = collection.count_documents({"Label": 1})

        logger.info(f"ETL complete. Total: {total_inserted:,}")
        logger.info(
            f"Label distribution  "
            f"BENIGN: {benign:,} | ATTACK: {attacks:,}"
        )

        print(f"\nETL Complete.")
        print(f"   Total records : {total_inserted:,}")
        print(f"   BENIGN  (0)   : {benign:,}")
        print(f"   ATTACK  (1)   : {attacks:,}")
        print(f"   Attack ratio  : {attacks/total_inserted*100:.1f}%")
        print(f"\n   Estimated MongoDB storage: ~"
              f"{total_inserted * 200 / (1024*1024):.0f} MB")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    main()