import pandas as pd

# Load your JSON
df = pd.read_json("cs_papers_100k_with_institution.json")

# Filter papers that have a primary institution
df_with_institution = df[df['primary_institution'].notna() & (df['primary_institution'] != '')]

print("Total papers:", len(df))
print("Papers with institution:", len(df_with_institution))

# Save to a new JSON
df_with_institution.to_json("papers_with_institution_filtered.json", orient='records', lines=False)
