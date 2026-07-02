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