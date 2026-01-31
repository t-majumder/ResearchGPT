import os
import yaml
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

HYPERPARAM_CONFIG_PATH = os.path.join(CONFIG_DIR, "hyperparams.yaml") # rag hyperparameters
PROMPT_CONFIG_PATH = os.path.join(CONFIG_DIR, "prompt.yaml") # prompt templates
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "models.yaml") # available models

# Load hyperparameter configurations
with open(HYPERPARAM_CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

PDFS_DIRECTORY = os.path.join(DATA_DIR, CONFIG["pdfs_directory"])
DB_PATH = os.path.join(DATA_DIR, CONFIG["db_path"])

EMBEDDING_MODEL = CONFIG["embedding_model"]

CHUNK_SIZE = CONFIG["text_splitting"]["chunk_size"]
CHUNK_OVERLAP = CONFIG["text_splitting"]["chunk_overlap"]
ADD_START_INDEX = CONFIG["text_splitting"]["add_start_index"]

SEARCH_TYPE = CONFIG["retrieval"]["search_type"]
TOP_K_RESULTS = int(CONFIG["retrieval"]["top_k"])
FETCH_K = int(CONFIG["retrieval"].get("fetch_k", max(20, TOP_K_RESULTS * 5)))
MIN_SIMILARITY = float(CONFIG["retrieval"].get("min_similarity", 0.0))

USE_RERANKER = bool(CONFIG["retrieval"].get("use_reranker", False))
RERANKER_MODEL = CONFIG["retrieval"].get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(CONFIG["retrieval"].get("rerank_top_n", TOP_K_RESULTS))

MAX_FILE_SIZE_BYTES = int(CONFIG["file_upload_limits"]["max_file_size_bytes"])
MAX_HISTORY_MESSAGES = int(CONFIG["chat_history"]["max_history_messages"])

# Loading Prompt configurations
with open(PROMPT_CONFIG_PATH, "r") as f:
    prompt_config = yaml.safe_load(f)
PROMPT_BEHAVIOR_WITH_RAG = prompt_config["prompts"]["behavior_with_rag"]
PROMPT_BEHAVIOR_WITHOUT_RAG = prompt_config["prompts"]["behavior_without_rag"]

# Loading Model configurations
with open(MODEL_CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)
AVAILABLE_MODELS = model_config["models"]