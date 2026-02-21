import pandas as pd
import requests
import country_converter as coco
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = "cs_papers_100k.json"
OUTPUT_FILE = "cs_papers_100k_with_institution.json"

OPENALEX_API_KEY = "fqKIe4pVlZVFTlh4pm9O3R"
OPENALEX_EMAIL = "soundariyavijayakumar22@gmail.com"

MAX_WORKERS = 16
SAVE_EVERY = 5000   # checkpoint

cc = coco.CountryConverter()


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_json(INPUT_FILE)

df["institutions"] = [[] for _ in range(len(df))]
df["primary_institution"] = None
df["region"] = None


# -----------------------------
# OpenAlex API
# -----------------------------
def fetch_openalex(doi):
    try:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        headers = {
            "Authorization": f"Bearer {OPENALEX_API_KEY}",
            "mailto": OPENALEX_EMAIL,
        }

        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        return r.json()
    except:
        return None


def process_row(idx, doi):
    if not doi or pd.isna(doi):
        return idx, [], None, None

    data = fetch_openalex(doi)
    if not data:
        return idx, [], None, None

    institutions = []
    countries = []

    for authorship in data.get("authorships", []):
        for inst in authorship.get("institutions", []):
            if inst.get("display_name"):
                institutions.append(inst["display_name"])
            if inst.get("country_code"):
                countries.append(inst["country_code"])

    institutions = list(set(institutions))
    countries = list(set(countries))

    primary_inst = institutions[0] if institutions else None
    primary_country = countries[0] if countries else None

    region = None
    if primary_country:
        region = cc.convert(primary_country, to="continent")

    return idx, institutions, primary_inst, region


# -----------------------------
# Parallel processing
# -----------------------------
print(f"Using {MAX_WORKERS} workers")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []

    for idx, row in df.iterrows():
        futures.append(
            executor.submit(process_row, idx, row["doi"])
        )

    completed = 0

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing papers"):
        idx, inst, primary, region = future.result()

        df.at[idx, "institutions"] = inst
        df.at[idx, "primary_institution"] = primary
        df.at[idx, "region"] = region

        completed += 1

        if completed % SAVE_EVERY == 0:
            df.to_json(OUTPUT_FILE, orient="records")
            print(f"Checkpoint saved at {completed}")


# -----------------------------
# Final save
# -----------------------------
df.to_json(OUTPUT_FILE, orient="records")
print(" Finished successfully!")
