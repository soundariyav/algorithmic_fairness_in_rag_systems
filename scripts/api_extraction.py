"""
Fetch 100K CS papers with institution data from OpenAlex using PyAlex.
Fix: over-fetch to account for papers dropped due to missing institutions.
Output: cs_papers_with_institutions.json saved to Google Drive
"""


!pip install pyalex tqdm -q

from pyalex import Works, config
from google.colab import drive
import json, time, os
from tqdm import tqdm


drive.mount('/content/drive')


PROJECT_DIR = "/content/drive/MyDrive/arxiv-rag-project"
os.makedirs(PROJECT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(PROJECT_DIR, "cs_papers_100k_with_institutions_final1.json")

TARGET      = 100_000

FETCH_LIMIT = 110_000
PER_PAGE    = 200       # 
config.email = "John.doe21@example.com"


REGION_MAP = {
    "US": "America", "CA": "America", "BR": "America", "MX": "America",
    "AR": "America", "CL": "America", "CO": "America",
    "GB": "Europe",  "DE": "Europe",  "FR": "Europe",  "IT": "Europe",
    "ES": "Europe",  "NL": "Europe",  "SE": "Europe",  "CH": "Europe",
    "PL": "Europe",  "RU": "Europe",  "PT": "Europe",  "BE": "Europe",
    "AT": "Europe",  "DK": "Europe",  "NO": "Europe",  "FI": "Europe",
    "GR": "Europe",  "CZ": "Europe",  "HU": "Europe",  "RO": "Europe",
    "CN": "Asia",    "IN": "Asia",    "JP": "Asia",    "KR": "Asia",
    "SG": "Asia",    "TW": "Asia",    "HK": "Asia",    "IL": "Asia",
    "IR": "Asia",    "PK": "Asia",    "SA": "Asia",    "TR": "Asia",
    "AU": "Oceania", "NZ": "Oceania",
    "ZA": "Africa",  "NG": "Africa",  "EG": "Africa",  "KE": "Africa",
    "ET": "Africa",  "GH": "Africa",  "TN": "Africa",  "MA": "Africa",
}

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))

def get_region(country_code):
    if not country_code:
        return "Unknown"
    return REGION_MAP.get(country_code.upper(), "Other")

def extract_paper(work):
    primary_institution = None
    primary_country     = None
    region              = "Unknown"

    for authorship in (work.get("authorships") or []):
        insts = authorship.get("institutions") or []
        if insts:
            inst = insts[0]
            primary_institution = inst.get("display_name")
            primary_country     = inst.get("country_code")
            region              = get_region(primary_country)
            break

    arxiv_id = None
    for loc in (work.get("locations") or []):
        src = loc.get("source") or {}
        if "arxiv" in (src.get("host_organization_name") or "").lower():
            landing = loc.get("landing_page_url") or ""
            if "arxiv.org/abs/" in landing:
                arxiv_id = landing.split("arxiv.org/abs/")[-1]
                break
    if not arxiv_id:
        arxiv_id = (work.get("ids") or {}).get("arxiv")

    return {
        "id":                  work.get("id"),
        "arxiv_id":            arxiv_id,
        "title":               (work.get("title") or "").strip(),
        "abstract":            reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_year":    work.get("publication_year"),
        "categories":          [t.get("display_name") for t in (work.get("topics") or [])[:3]],
        "primary_institution": primary_institution,
        "primary_country":     primary_country,
        "region":              region,
        "cited_by_count":      work.get("cited_by_count", 0),
        "doi":                 work.get("doi"),
    }


def fetch_papers(target=TARGET, fetch_limit=FETCH_LIMIT):
    papers    = []
    seen_ids  = set()
    fetched   = 0      

    query = (
        Works()
        .filter(
            concepts={"wikidata": "https://www.wikidata.org/wiki/Q21198"},
            has_abstract=True,
            locations={"source": {"id": "https://openalex.org/S4306400194"}},
        )
        .select([
            "id", "title", "abstract_inverted_index", "publication_year",
            "authorships", "topics", "cited_by_count",
            "doi", "ids", "locations"
        ])
    )


    pbar = tqdm(total=target, desc="Papers with institutions")

    for page in query.paginate(per_page=PER_PAGE, n_max=fetch_limit):
        for work in page:
            wid = work.get("id")
            if wid in seen_ids:
                continue
            seen_ids.add(wid)
            fetched += 1

            paper = extract_paper(work)

            if paper["primary_institution"]:
                papers.append(paper)
                pbar.update(1)

            if len(papers) >= target:
                pbar.close()
                print(f"\nFetched {fetched:,} raw works to collect {len(papers):,} with institutions "
                      f"(dropout rate: {1 - len(papers)/fetched:.1%})")
                return papers[:target]

        time.sleep(0.1)

    pbar.close()
    print(f"\nFetched {fetched:,} raw works; collected {len(papers):,} with institutions "
          f"(dropout rate: {1 - len(papers)/fetched:.1%})")

    if len(papers) < target:
        print(f" Only {len(papers):,} papers found — OpenAlex may not have {target:,} "
              f"CS arXiv papers with institution data. Consider raising FETCH_LIMIT further.")

    return papers


print(f"Starting fetch — target: {TARGET:,} papers with institutions (over-fetching up to {FETCH_LIMIT:,})...")
papers = fetch_papers()

print(f"\nFinal count: {len(papers):,} papers")


from collections import Counter
regions = Counter(p["region"] for p in papers)
print("\nRegion distribution:")
for region, count in regions.most_common():
    print(f"  {region:<12} {count:>6,}  ({count/len(papers):.1%})")


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"\nSaved → {OUTPUT_FILE}")
print("Done!")
