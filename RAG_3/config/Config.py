from pathlib import Path
import yaml
import os
from dotenv import load_dotenv
load_dotenv()
# Đường dẫn đến config.yaml
CONFIG_PATH = Path(__file__).parent / "config.yaml"
def load_config():
    #Đọc file config.yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load một lần duy nhất
config = load_config()

config["key"]["HF_TOKEN"] = os.getenv("HF_TOKEN")
config["key"]["GEMINI_KEY"] = os.getenv("GEMINI_KEY")
# ==========================
# Chunking
# ==========================
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]
MAX_CHUNK_SIZE = config["chunking"]["max_chunk_size"]
MIN_CHUNK_SIZE = config["chunking"]["min_chunk_size"]
THRESHOLD = config["chunking"]["threshold"]

# ==========================
# Retrieval
# ==========================
TOP_K = config["retrieval"]["top_k"]

# ==========================
# Generation
# ==========================
TEMPERATURE = config["generation"]["temperature"]
MAX_TOKENS = config["generation"]["max_tokens"]

# ==========================
# Key
# ==========================
HF_TOKEN = config["key"]["HF_TOKEN"]
GEMINI_KEY = config["key"]["GEMINI_KEY"]

# ==========================
# MODEL
# ==========================
MODEL_NAME = config["model"]["MODEL_NAME"]

# ==========================
# VECTOR STORE
# ==========================
VECTOR_DB_TYPE = config.get("vectorstore", {}).get("db_type", "chroma")
persist_dir = config.get("vectorstore", {}).get("persist_directory", "data/vector_store")
VECTOR_DB_PERSIST_DIR = str(Path(__file__).parent.parent / persist_dir)
VECTOR_DB_COLLECTION = config.get("vectorstore", {}).get("collection_name", "rag_collection")
VECTOR_DB_SIZE = config.get("vectorstore", {}).get("vector_size", 1024)