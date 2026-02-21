import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = 'papers_with_institution_filtered.json'
# --- Load JSON file ---
df = pd.read_json(file_path)

# --- Quick checks ---
print("Total papers:", len(df))
print("Papers with institutions:", df['primary_institution'].notna().sum())

# --- Clean data ---
# Replace None with 'Unknown' for region
df['region'] = df['region'].fillna('Unknown')
df['primary_institution'] = df['primary_institution'].fillna('Unknown')

# --- Count papers per region ---
region_counts = df['region'].value_counts()
print(region_counts)

# --- Count top institutions globally ---
top_institutions = df['primary_institution'].value_counts().head(20)
print(top_institutions)

# --- Visualization 1: Bar chart of papers per region ---
plt.figure(figsize=(8,8))
plt.pie(
    region_counts.values,      # sizes
    labels=region_counts.index, # labels
    autopct='%1.1f%%',         # show percentages
    startangle=140             # rotate for better layout
)
plt.title("Distribution of Papers by Region")
plt.tight_layout()
plt.savefig("region_distribution_pie.png")
plt.show()

# --- Visualization 2: Top 20 institutions ---
plt.figure(figsize=(12,6))
sns.barplot(x=top_institutions.values, y=top_institutions.index, palette="magma")
plt.title("Top 20 Institutions by Paper Count")
plt.xlabel("Number of Papers")
plt.ylabel("Institution")
plt.tight_layout()
plt.savefig("top_institutions.png")
plt.show()
