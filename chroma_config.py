import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_storage"

client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="fair_rag_collection",
    embedding_function=embedding_function
)
