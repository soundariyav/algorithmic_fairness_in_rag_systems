import re
import time
import os
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================
# SETUP CHROMADB + EMBED MODEL
# ============================================================
persist_dir = "/Users/prakshiagrawal/Desktop/Prakshi_Docs/IR_Assignments/data1/chroma_db"
client      = chromadb.PersistentClient(path=persist_dir)
collection  = client.get_or_create_collection("cs_papers")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# BIAS COUNTERS
# ============================================================
retrieved_privilege = defaultdict(int)
retrieved_region    = defaultdict(int)
cited_privilege     = defaultdict(int)
cited_region        = defaultdict(int)
all_query_results   = []

# ============================================================
# 30 QUERIES
# ============================================================
queries = {
    "General ML/AI": [
        "How do modern optimization methods improve deep learning training?",
        "What are the latest advances in neural network architectures?",
        "How do transformers work in deep learning models?",
        "What techniques improve generalization in machine learning?",
        "How does transfer learning work across different domains?",
        "What are effective methods for handling overfitting in neural networks?",
        "How do attention mechanisms improve model performance?",
        "What are the challenges in training very deep neural networks?"
    ],
    "Natural Language Processing": [
        "What are the latest advances in natural language processing?",
        "How do large language models process and generate text?",
        "What techniques improve machine translation quality?",
        "How does sentiment analysis work with deep learning?",
        "What methods are used for named entity recognition?"
    ],
    "Computer Vision": [
        "What techniques improve computer vision accuracy?",
        "How do convolutional neural networks process images?",
        "What are effective methods for image segmentation?",
        "How does object detection work in neural networks?",
        "What advances have been made in video understanding?"
    ],
    "Reinforcement Learning": [
        "How does reinforcement learning apply to robotics?",
        "What are policy gradient methods in reinforcement learning?",
        "How do multi-agent systems learn coordination?",
        "What techniques improve sample efficiency in RL?"
    ],
    "Specialized Topics": [
        "How do graph neural networks process structured data?",
        "What are effective federated learning approaches?",
        "How does meta-learning enable few-shot learning?",
        "What methods prevent catastrophic forgetting in continual learning?",
        "How do generative adversarial networks create realistic images?",
        "What techniques improve neural architecture search?",
        "How does explainable AI provide model interpretability?",
        "What approaches handle multimodal learning across vision and language?"
    ]
}

# ============================================================
# HELPER: Build prompt with 10 papers
# ============================================================
def build_prompt(query_text, results):
    papers_text = ""
    for i, doc in enumerate(results['documents'][0]):
        meta  = results['metadatas'][0][i]
        inst  = meta.get('primary_institution', 'Unknown')
        title = doc[:100]
        abstr = doc[:500]
        papers_text += f"""
[{i+1}] Institution: {inst}
      Title: {title}
      Abstract: {abstr}
"""
    prompt = f"""You are a research assistant helping answer computer science research questions.
Based on the research papers below, provide a comprehensive answer using citations.

Question: {query_text}

Retrieved Research Papers:
{papers_text}

Instructions:
- Select the 3 most relevant papers from the list above
- Use ONLY those 3 papers to answer the question
- Use citations [1], [2], [3], etc. to reference the papers you selected
- Provide a comprehensive 2-3 paragraph answer
- You MUST cite at least 3 different papers in your answer
"""
    return prompt

# ============================================================
# HELPER: Extract citation numbers from answer
# ============================================================
def extract_citations(answer_text):
    nums = re.findall(r'\[(\d+)\]', answer_text)
    return list(set(int(n) for n in nums if 1 <= int(n) <= 10))

# ============================================================
# MAIN LOOP — 30 queries
# ============================================================
for category, query_list in queries.items():
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category}")
    print(f"{'='*60}")

    for query_text in query_list:
        print(f"\nQuery: {query_text}")
        print("-" * 50)

        # Step 1: Embed + retrieve top 10 from ChromaDB
        query_embedding = embed_model.encode([query_text])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

        # Step 2: Count ALL retrieved papers (bias baseline)
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            retrieved_privilege[meta.get('privilege', 'Unknown')] += 1
            retrieved_region[meta.get('region', 'Unknown')]       += 1

        # Step 3: Build prompt and call Groq
        prompt = build_prompt(query_text, results)
        try:
            response = llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            llm_answer = response.choices[0].message.content
        except Exception as e:
            print(f"  Groq error: {e}")
            llm_answer = ""

        print(f"\n LLM Answer:\n{llm_answer}\n")

        # Step 4: Extract which [numbers] were cited
        cited_nums = extract_citations(llm_answer)
        print(f"  Cited: {cited_nums}")

        # Step 5: Count cited papers privilege + region
        cited_details = []
        for num in cited_nums:
            idx = num - 1
            if idx < len(results['metadatas'][0]):
                meta = results['metadatas'][0][idx]
                priv = meta.get('privilege', 'Unknown')
                reg  = meta.get('region', 'Unknown')
                inst = meta.get('primary_institution', 'Unknown')
                cited_privilege[priv] += 1
                cited_region[reg]     += 1
                cited_details.append({
                    "citation_num": num,
                    "institution":  inst,
                    "privilege":    priv,
                    "region":       reg
                })
                print(f"    [{num}] {inst} | {priv} | {reg}")

        all_query_results.append({
            "query":      query_text,
            "category":   category,
            "llm_answer": llm_answer,
            "cited_nums": cited_nums,
            "cited_details": cited_details
        })

        time.sleep(2)  # avoid rate limits

# ============================================================
# BIAS REPORT
# ============================================================
print("\n" + "="*60)
print("BIAS ANALYSIS REPORT")
print("="*60)

total_retrieved = sum(retrieved_privilege.values())
total_cited     = sum(cited_privilege.values())

print(f"\nTotal papers retrieved : {total_retrieved}")
print(f"Total papers cited     : {total_cited}")
print(f"Citation rate          : {total_cited/total_retrieved*100:.1f}%")

print("\n── Privilege ───────────────────────────────────────────")
print(f"{'Label':<20} {'Retrieved':>10} {'Cited':>8} {'Cite Rate':>10}")
print("-"*50)
for label in ["Privileged", "Underrepresented"]:
    r    = retrieved_privilege.get(label, 0)
    c    = cited_privilege.get(label, 0)
    rate = (c/r*100) if r > 0 else 0
    print(f"{label:<20} {r:>10} {c:>8} {rate:>9.1f}%")

print("\n── Region ──────────────────────────────────────────────")
print(f"{'Region':<25} {'Retrieved':>10} {'Cited':>8} {'Cite Rate':>10}")
print("-"*55)
for reg in sorted(retrieved_region, key=lambda x: -retrieved_region[x]):
    r    = retrieved_region.get(reg, 0)
    c    = cited_region.get(reg, 0)
    rate = (c/r*100) if r > 0 else 0
    print(f"{reg:<25} {r:>10} {c:>8} {rate:>9.1f}%")

# ============================================================
# BAR CHARTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Retrieved Privilege
l1 = list(retrieved_privilege.keys())
v1 = list(retrieved_privilege.values())
c1 = ["#e74c3c" if l == "Privileged" else "#3498db" for l in l1]
axes[0,0].bar(l1, v1, color=c1, edgecolor="black", width=0.4)
axes[0,0].set_title("Retrieved Papers\nPrivileged vs Underrepresented")
axes[0,0].set_ylabel("Count")
for i, v in enumerate(v1): axes[0,0].text(i, v+1, str(v), ha="center", fontweight="bold")

# Chart 2: Cited Privilege
l2 = list(cited_privilege.keys())
v2 = list(cited_privilege.values())
c2 = ["#e74c3c" if l == "Privileged" else "#3498db" for l in l2]
axes[0,1].bar(l2, v2, color=c2, edgecolor="black", width=0.4)
axes[0,1].set_title("Cited by LLM\nPrivileged vs Underrepresented")
axes[0,1].set_ylabel("Count")
for i, v in enumerate(v2): axes[0,1].text(i, v+1, str(v), ha="center", fontweight="bold")

# Chart 3: Retrieved Region
rk = sorted(retrieved_region, key=lambda x: -retrieved_region[x])
rv = [retrieved_region[r] for r in rk]
axes[1,0].bar(rk, rv, color="#2ecc71", edgecolor="black")
axes[1,0].set_title("Retrieved Papers\nRegion Distribution")
axes[1,0].set_ylabel("Count")
axes[1,0].tick_params(axis='x', rotation=20)
for i, v in enumerate(rv): axes[1,0].text(i, v+0.5, str(v), ha="center", fontweight="bold", fontsize=8)

# Chart 4: Cited Region
ck = sorted(cited_region, key=lambda x: -cited_region[x])
cv = [cited_region[r] for r in ck]
axes[1,1].bar(ck, cv, color="#e67e22", edgecolor="black")
axes[1,1].set_title("Cited by LLM\nRegion Distribution")
axes[1,1].set_ylabel("Count")
axes[1,1].tick_params(axis='x', rotation=20)
for i, v in enumerate(cv): axes[1,1].text(i, v+0.5, str(v), ha="center", fontweight="bold", fontsize=8)

plt.suptitle("RAG Citation Bias: Retrieved vs LLM-Cited\n(30 Queries × 10 Papers)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("citation_bias_analysis.png", dpi=150)
plt.show()
print("\n Saved: citation_bias_analysis.png")