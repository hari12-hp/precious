# ------------------ Keras-3 / Transformers compatibility ------------------
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"       # NEW: must precede TF/Transformers

# launcher_pc_optimized.py
import sys
import json
import time
import hashlib
import sqlite3
import logging
import traceback
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional OCR deps (loaded defensively)
try:
    from PIL import ImageGrab, Image
except ImportError:
    ImageGrab = None
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

# Ollama LLM integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

# ------------------------- GLOBAL CACHING -------------------------
_embed_model_cache: Optional[SentenceTransformer] = None
_embed_model_lock = threading.Lock()
_easyocr_reader = None

# ------------------------- LOGGING SETUP -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------- CONFIG -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "memory.db")
DOCS_PATH = os.path.join(BASE_DIR, "docs.json")
INDEX_PATH = os.path.join(BASE_DIR, "docs.index")
INDEX_TMP = INDEX_PATH + ".tmp"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Index selection and params
AUTO_HNSW = True
HNSW_AUTO_THRESHOLD = 5000
USE_HNSW = True
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# Conservative threading to avoid oversubscription
FAISS_THREADS = max(1, min(4, (os.cpu_count() or 1) // 2))

# OCR settings
ENABLE_OCR = True
OCR_ENGINE = "both"
EASYOCR_LANGS = ["en"]
EASYOCR_GPU = False
TESSERACT_CMD = None
OCR_MIN_CHARS = 24
OCR_TIMEOUT_S = 4.0
OCR_REGION = None
OCR_MAX_RECENT = 10  # Limit recent screen captures to prevent flooding

# Performance settings
CACHE_EMBEDDINGS = True  # Store embeddings to avoid re-encoding
PARALLEL_BATCH_THRESHOLD = 10  # Parallelize batch queries above this size

# Ollama + LoRA LLM settings
OLLAMA_BASE_MODEL = "deepseek-r1:1.5b"  # Base model
OLLAMA_LORA_MODEL = "deepseek-r1-rag"   # Your domain-specific LoRA model
USE_LORA_MODEL = False  # Set to True when you have a LoRA model ready
AUTO_FALLBACK = True    # Fallback to base model if LoRA fails

# LoRA model templates for better structured responses
LORA_SYSTEM_PROMPT = """You are a specialized AI assistant with domain expertise. Use the provided context to give accurate, detailed responses. Be concise but comprehensive."""

# ------------------------- UTILS -------------------------
def atomic_write_bytes(tmp_path: str, final_path: str, writer_fn) -> None:
    """Atomically write file using temporary file and replace."""
    writer_fn(tmp_path)
    os.replace(tmp_path, final_path)

def sha1_text(s: str) -> str:
    """Generate SHA1 hash of text for deduplication."""
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def now_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)

# ------------------------- ID ↔ TEXT MAP -------------------------
def id_to_text_map(store: Dict[str, Any]) -> Dict[int, str]:
    """Builds a mapping from doc_id → text for fast lookup."""
    return {item["id"]: item["text"] for item in store.get("docs", [])}

# ------------------------- LORA MODEL MANAGEMENT -------------------------
def check_ollama_model_exists(model_name: str) -> bool:
    """Check if a specific model exists in Ollama"""
    try:
        result = ollama.list()
        model_names = [model['name'] for model in result.get('models', [])]
        return any(model_name in name for name in model_names)
    except Exception as e:
        logger.debug(f"Failed to check model existence: {e}")
        return False

def create_rag_optimized_lora_model(base_model: str = OLLAMA_BASE_MODEL, 
                                   model_name: str = OLLAMA_LORA_MODEL) -> bool:
    """Create a RAG-optimized model configuration in Ollama"""
    try:
        # Create optimized Modelfile for RAG tasks
        modelfile_content = f"""FROM {base_model}

# RAG-optimized parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER top_k 25
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# RAG-specific system prompt
SYSTEM {LORA_SYSTEM_PROMPT}

# Template for structured RAG responses
TEMPLATE \"\"\"<|system|>
{LORA_SYSTEM_PROMPT}
<|/system|>

<|user|>
{{{{ .Prompt }}}}
<|/user|>

<|assistant|>
\"\"\"
"""
        
        modelfile_path = os.path.join(BASE_DIR, "Modelfile.rag")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        # Create model in Ollama
        result = subprocess.run([
            "ollama", "create", model_name, "-f", modelfile_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Successfully created RAG-optimized model: {model_name}")
            return True
        else:
            logger.warning(f"Failed to create RAG model: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating RAG-optimized model: {e}")
        return False

# ------------------------- ENHANCED LLM INTEGRATION -------------------------
def query_llm_with_rag(prompt: str, model: str = None, force_base: bool = False) -> str:
    """Send RAG-enhanced prompt to local LLM via Ollama with LoRA optimization"""
    if not OLLAMA_AVAILABLE:
        logger.warning("Ollama not available, returning prompt only")
        return f"[Ollama not installed] Would send this prompt:\n{prompt}"
    
    # Determine which model to use
    if force_base:
        selected_model = OLLAMA_BASE_MODEL
        use_structured = False
    elif model:
        selected_model = model
        use_structured = False
    elif USE_LORA_MODEL:
        selected_model = OLLAMA_LORA_MODEL
        use_structured = True
        
        # Check if LoRA model exists, create if not
        if not check_ollama_model_exists(OLLAMA_LORA_MODEL):
            logger.info(f"LoRA model {OLLAMA_LORA_MODEL} not found, creating...")
            if create_rag_optimized_lora_model():
                logger.info("RAG-optimized model created successfully")
            else:
                logger.warning("Failed to create RAG model, falling back to base")
                selected_model = OLLAMA_BASE_MODEL
                use_structured = False
    else:
        selected_model = OLLAMA_BASE_MODEL
        use_structured = False
    
    try:
        logger.info(f"Querying {selected_model} ({'structured' if use_structured else 'standard'} prompt)")
        
        if use_structured:
            # Parse memory and query from prompt for structured format
            if 'MEMORY:' in prompt and 'USER QUERY:' in prompt:
                memory_part = prompt.split('MEMORY:')[1].split('USER QUERY:')[0].strip()
                query_part = prompt.split('USER QUERY:')[1].replace('Answer concisely:', '').strip()
                
                structured_prompt = f"""Context Information:
{memory_part}

Question: {query_part}

Instructions: Using the context provided above, answer the question accurately and concisely. If the context doesn't contain relevant information, say so clearly."""
            else:
                structured_prompt = prompt
            
            response = ollama.chat(
                model=selected_model,
                messages=[{
                    'role': 'user', 
                    'content': structured_prompt
                }],
                options={
                    'temperature': 0.3,    # Lower temperature for more focused responses
                    'top_p': 0.85,
                    'top_k': 25,
                    'repeat_penalty': 1.1,
                    'num_ctx': 4096
                }
            )
        else:
            # Standard prompt for base models
            response = ollama.chat(
                model=selected_model,
                messages=[{
                    'role': 'user', 
                    'content': prompt
                }],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
        
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"LLM query failed with {selected_model}: {e}")
        
        # Fallback logic
        if AUTO_FALLBACK and not force_base and selected_model != OLLAMA_BASE_MODEL:
            logger.info("Falling back to base model...")
            return query_llm_with_rag(prompt, force_base=True)
        
        return f"Error querying LLM: {e}\n\nOriginal prompt was:\n{prompt}"

# ------------------------- EMBEDDING MODEL CACHE -------------------------
def get_cached_embed_model(name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Get cached embedding model to avoid reloading on each run."""
    global _embed_model_cache
    with _embed_model_lock:
        if _embed_model_cache is None:
            logger.info(f"Loading embedding model: {name}")
            _embed_model_cache = SentenceTransformer(name)
        return _embed_model_cache

def get_embed_dim(embed_model: SentenceTransformer) -> int:
    """Robustly derive embedding dimension from the model."""
    dim_func = getattr(embed_model, "get_sentence_embedding_dimension", None)
    if callable(dim_func):
        return embed_model.get_sentence_embedding_dimension()
    # Fallback: encode a dummy string
    emb = embed_model.encode(["x"], convert_to_numpy=True, normalize_embeddings=True)
    return int(emb.shape[1])

def encode_texts(embed_model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts to normalized embeddings - ensures normalize_embeddings=True for cosine."""
    if not texts:
        return np.array([]).reshape(0, get_embed_dim(embed_model)).astype("float32")
    
    emb = embed_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True  # Critical for IndexFlatIP cosine similarity
    ).astype("float32")
    
    # Double-check normalization (defensive programming)
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        logger.warning("Embeddings not properly normalized, fixing...")
        emb = emb / norms[:, np.newaxis]
    
    return emb

# ------------------------- SQLITE MEMORY -------------------------
def init_sqlite(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with optimized settings."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=536870912")  # 512MB
    conn.execute("PRAGMA foreign_keys=ON")
    
    # Create tables
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)")
    
    # Optional: store embeddings in SQLite to avoid re-encoding
    if CACHE_EMBEDDINGS:
        conn.execute("""CREATE TABLE IF NOT EXISTS doc_embeddings (
            doc_id INTEGER PRIMARY KEY,
            embedding BLOB,
            model_name TEXT,
            created_at INTEGER
        )""")
    
    return conn

def seed_facts(conn: sqlite3.Connection) -> None:
    """Seed database with initial facts if empty."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM facts")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO facts (content) VALUES (?)",
            ("DeepSeek R1 is a 1.5B parameter LLM optimized for low VRAM.",),
        )
        conn.commit()

def get_all_facts(conn: sqlite3.Connection) -> List[str]:
    """Retrieve all facts from database."""
    cur = conn.cursor()
    cur.execute("SELECT content FROM facts")
    return [row[0] for row in cur.fetchall()]

# ------------------------- ENHANCED DOC STORE -------------------------
def _default_store() -> Dict[str, Any]:
    """Create default empty document store with recent screen captures tracking."""
    return {
        "next_id": 0, 
        "docs": [], 
        "hash_to_id": {},
        "recent_screen_hashes": [],  # Track recent screen captures
        "embeddings": {}  # Cache embeddings if CACHE_EMBEDDINGS
    }

def load_store(path: str) -> Dict[str, Any]:
    """Load document store from JSON file."""
    if not os.path.exists(path):
        return _default_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Migrate legacy formats
        if "hash_to_id" not in data:
            data["hash_to_id"] = {}
            for item in data.get("docs", []):
                h = sha1_text(item["text"].strip().lower())
                data["hash_to_id"][h] = item["id"]
        
        if "recent_screen_hashes" not in data:
            data["recent_screen_hashes"] = []
        
        if "embeddings" not in data and CACHE_EMBEDDINGS:
            data["embeddings"] = {}
            
        return data
    except Exception as e:
        logger.error(f"Failed to load store from {path}: {e}")
        return _default_store()

def save_store(store: Dict[str, Any], path: str) -> None:
    """Save document store to JSON file atomically."""
    tmp = path + ".tmp"
    def _writer(p):
        # Don't save embeddings to JSON (too large), use SQLite instead
        store_copy = store.copy()
        if "embeddings" in store_copy:
            store_copy.pop("embeddings")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(store_copy, f, ensure_ascii=False)
    atomic_write_bytes(tmp, path, _writer)

# ------------------------- EMBEDDING CACHE MANAGEMENT -------------------------
def save_embeddings_to_db(conn: sqlite3.Connection, doc_id: int, embedding: np.ndarray, 
                         model_name: str) -> None:
    """Save embedding to SQLite for faster rebuilds."""
    if not CACHE_EMBEDDINGS:
        return
    try:
        cur = conn.cursor()
        embedding_blob = embedding.tobytes()
        cur.execute("""INSERT OR REPLACE INTO doc_embeddings 
                      (doc_id, embedding, model_name, created_at) 
                      VALUES (?, ?, ?, ?)""",
                   (doc_id, embedding_blob, model_name, now_ms()))
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to cache embedding for doc {doc_id}: {e}")

def load_embeddings_from_db(conn: sqlite3.Connection, doc_ids: List[int], 
                          model_name: str) -> Dict[int, np.ndarray]:
    """Load cached embeddings from SQLite."""
    if not CACHE_EMBEDDINGS or not doc_ids:
        return {}
    
    try:
        cur = conn.cursor()
        placeholders = ",".join("?" * len(doc_ids))
        cur.execute(f"""SELECT doc_id, embedding FROM doc_embeddings 
                       WHERE doc_id IN ({placeholders}) AND model_name = ?""",
                   doc_ids + [model_name])
        
        cached = {}
        embed_dim = None
        for doc_id, blob in cur.fetchall():
            try:
                if embed_dim is None:
                    # Determine embedding dimension from first result
                    test_arr = np.frombuffer(blob, dtype=np.float32)
                    embed_dim = len(test_arr)
                
                embedding = np.frombuffer(blob, dtype=np.float32).reshape(1, embed_dim)
                cached[doc_id] = embedding[0]  # Store as 1D array
            except Exception as e:
                logger.warning(f"Failed to deserialize embedding for doc {doc_id}: {e}")
        
        logger.info(f"Loaded {len(cached)}/{len(doc_ids)} cached embeddings")
        return cached
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings: {e}")
        return {}

# ------------------------- FAISS INDEX WITH CACHING -------------------------
def set_faiss_threads(n_threads: int = FAISS_THREADS) -> None:
    """Set FAISS thread count conservatively."""
    faiss.omp_set_num_threads(int(n_threads))

def _wrap_with_idmap(base_index) -> Any:
    """Wrap FAISS index with ID mapping for stable external IDs."""
    if hasattr(faiss, "IndexIDMap2"):
        return faiss.IndexIDMap2(base_index)
    return faiss.IndexIDMap(base_index)

def _pick_use_hnsw(num_docs: int) -> bool:
    """Switch to HNSW if corpus is large for faster search."""
    if AUTO_HNSW:
        return num_docs >= HNSW_AUTO_THRESHOLD
    return USE_HNSW

def build_index_from_embeddings_cached(store: Dict[str, Any], embed_model: SentenceTransformer,
                                     conn: sqlite3.Connection) -> Tuple[Any, List[int]]:
    """Build FAISS index using cached embeddings when possible."""
    if not store["docs"]:
        d = get_embed_dim(embed_model)
        base = faiss.IndexFlatIP(d)
        return _wrap_with_idmap(base), []
    
    doc_ids = [item["id"] for item in store["docs"]]
    texts = [item["text"] for item in store["docs"]]
    
    # Try to load cached embeddings
    cached_embeddings = load_embeddings_from_db(conn, doc_ids, EMBED_MODEL_NAME)
    
    # Identify which docs need encoding
    missing_ids = [doc_id for doc_id in doc_ids if doc_id not in cached_embeddings]
    missing_texts = [texts[i] for i, doc_id in enumerate(doc_ids) if doc_id in missing_ids]
    
    # Encode missing texts
    if missing_texts:
        logger.info(f"Encoding {len(missing_texts)} missing embeddings")
        missing_embeddings = encode_texts(embed_model, missing_texts)
        
        # Cache new embeddings
        for i, doc_id in enumerate([doc_id for doc_id in doc_ids if doc_id in missing_ids]):
            cached_embeddings[doc_id] = missing_embeddings[i]
            save_embeddings_to_db(conn, doc_id, missing_embeddings[i], EMBED_MODEL_NAME)
    
    # Assemble all embeddings in correct order
    all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in doc_ids])
    
    # Build index
    use_hnsw = _pick_use_hnsw(len(doc_ids))
    d = int(all_embeddings.shape[1])
    
    if use_hnsw:
        base = faiss.IndexHNSWFlat(d, HNSW_M)
        base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        base.hnsw.efSearch = HNSW_EF_SEARCH
        logger.info(f"Built HNSW index with d={d}, M={HNSW_M}")
    else:
        base = faiss.IndexFlatIP(d)
        logger.info(f"Built Flat index with d={d}")
    
    index = _wrap_with_idmap(base)
    ids_array = np.array(doc_ids, dtype=np.int64)
    index.add_with_ids(all_embeddings, ids_array)
    
    return index, doc_ids

# ------------------------- OCR WITH PROPER TIMEOUT -------------------------
def get_easyocr_reader() -> Optional[Any]:
    """Get EasyOCR reader instance with lazy initialization."""
    global _easyocr_reader
    if _easyocr_reader is None and easyocr is not None:
        try:
            _easyocr_reader = easyocr.Reader(EASYOCR_LANGS, gpu=EASYOCR_GPU)
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            return None
    return _easyocr_reader

def configure_tesseract_path() -> None:
    """Configure Tesseract executable path if needed."""
    if pytesseract is None:
        return
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def _ocr_tesseract(img) -> str:
    """Run Tesseract OCR on image."""
    if pytesseract is None:
        return ""
    try:
        cfg = r"--oem 3 --psm 6"
        return pytesseract.image_to_string(img, config=cfg)
    except Exception as e:
        logger.debug(f"Tesseract OCR failed: {e}")
        return ""

def _ocr_easyocr(img) -> str:
    """Run EasyOCR on image."""
    if easyocr is None:
        return ""
    try:
        reader = get_easyocr_reader()
        if reader is None:
            return ""
        np_img = np.array(img)
        results = reader.readtext(np_img, detail=0)
        return " ".join(results) if results else ""
    except Exception as e:
        logger.debug(f"EasyOCR failed: {e}")
        return ""

def screen_to_text_with_timeout(region: Optional[Tuple[int, int, int, int]] = None,
                               screen_path: Optional[str] = None) -> str:
    """Capture screen and extract text using OCR with proper timeout handling."""
    if not ENABLE_OCR:
        return ""

    configure_tesseract_path()

    # Load image from file or capture screen
    img = None
    if screen_path and os.path.exists(screen_path):
        try:
            img = Image.open(screen_path)
            logger.debug(f"Loaded image from {screen_path}")
        except Exception as e:
            logger.debug(f"Failed to load image from {screen_path}: {e}")
    
    if img is None:
        try:
            if ImageGrab is not None:
                img = ImageGrab.grab(bbox=region)
        except Exception as e:
            logger.debug(f"Screen grab failed: {e}")
            # Fallback to clipboard
            try:
                if ImageGrab is not None:
                    img = ImageGrab.grabclipboard()
            except Exception as e2:
                logger.debug(f"Clipboard grab also failed: {e2}")
                return ""

    if img is None:
        logger.debug("No image captured (likely Wayland/macOS permission issue)")
        return ""

    # Use ThreadPoolExecutor for proper timeout control
    texts = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        if OCR_ENGINE in ("tesseract", "both"):
            futures.append(("tesseract", executor.submit(_ocr_tesseract, img)))
        
        if OCR_ENGINE in ("easyocr", "both"):
            futures.append(("easyocr", executor.submit(_ocr_easyocr, img)))
        
        # Collect results with timeout
        for name, future in futures:
            try:
                result = future.result(timeout=OCR_TIMEOUT_S)
                if result:
                    texts.append(result)
                    logger.debug(f"{name} OCR succeeded")
            except FutureTimeoutError:
                logger.debug(f"{name} OCR timed out after {OCR_TIMEOUT_S}s")
                future.cancel()
            except Exception as e:
                logger.debug(f"{name} OCR failed: {e}")

    out = "\n".join([t for t in texts if t]).strip()
    return out if len(out.strip()) >= OCR_MIN_CHARS else ""

def add_screen_memory_smart(store: Dict[str, Any], index: Any, 
                          embed_model: SentenceTransformer, conn: sqlite3.Connection,
                          screen_path: Optional[str] = None) -> Tuple[Dict[str, Any], Any]:
    """Capture screen and add OCR text to memory with smart deduplication."""
    screen_text = screen_to_text_with_timeout(region=OCR_REGION, screen_path=screen_path)
    if not screen_text:
        return store, index
    
    # Smart deduplication for screen captures
    text_hash = sha1_text(screen_text.strip().lower())
    
    # Check if this screen capture is too recent/similar
    if text_hash in store.get("recent_screen_hashes", []):
        logger.debug("Skipping duplicate screen capture")
        return store, index
    
    # Add to recent screens and limit size
    if "recent_screen_hashes" not in store:
        store["recent_screen_hashes"] = []
    
    store["recent_screen_hashes"].append(text_hash)
    if len(store["recent_screen_hashes"]) > OCR_MAX_RECENT:
        store["recent_screen_hashes"] = store["recent_screen_hashes"][-OCR_MAX_RECENT:]
    
    logger.info("Adding unique screen capture to memory")
    return add_docs_cached([screen_text], store, embed_model, index, conn, min_chars=OCR_MIN_CHARS)

# ------------------------- ENHANCED MUTATIONS -------------------------
def add_docs_cached(texts: List[str], store: Dict[str, Any], embed_model: SentenceTransformer, 
                   index: Any, conn: sqlite3.Connection, min_chars: int = 0) -> Tuple[Dict[str, Any], Any]:
    """Add new documents with embedding caching."""
    if not texts:
        return store, index
    
    new_items = []
    for t in texts:
        norm = (t or "").strip().lower()
        if len(norm) < min_chars:
            continue
        h = sha1_text(norm)
        if h in store["hash_to_id"]:
            continue
        new_items.append({"id": store["next_id"], "text": t})
        store["hash_to_id"][h] = store["next_id"]
        store["next_id"] += 1
    
    if not new_items:
        return store, index

    logger.info(f"Adding {len(new_items)} new documents")
    emb = encode_texts(embed_model, [x["text"] for x in new_items])
    ids = np.array([x["id"] for x in new_items], dtype=np.int64)
    
    # Cache embeddings
    for i, item in enumerate(new_items):
        save_embeddings_to_db(conn, item["id"], emb[i], EMBED_MODEL_NAME)
    
    index.add_with_ids(emb, ids)
    store["docs"].extend(new_items)
    return store, index

# ------------------------- ENHANCED RETRIEVAL -------------------------
def retrieve_top_k_enhanced(query: str, embed_model: SentenceTransformer, index: Any, 
                          id2text: Dict[int, str], k: int) -> List[str]:
    """Enhanced retrieval with better error handling and logging."""
    if not id2text:
        return []
    k = max(1, min(k, len(id2text)))
    q_emb = encode_texts(embed_model, [query])
    D, I = index.search(q_emb, k)
    
    ids = []
    mismatched_ids = []
    for x in I[0]:
        try:
            doc_id = int(x)
            if doc_id in id2text:
                ids.append(doc_id)
            else:
                mismatched_ids.append(doc_id)
        except (ValueError, TypeError):
            logger.debug(f"Invalid ID in search results: {x}")
    
    if mismatched_ids:
        logger.debug(f"Found {len(mismatched_ids)} mismatched IDs in search results")
    
    return [id2text[i] for i in ids]

def retrieve_batch_top_k_parallel(queries: List[str], embed_model: SentenceTransformer, 
                                 index: Any, id2text: Dict[int, str], k: int) -> List[List[str]]:
    """Batch retrieval with optional parallelization for large batches."""
    if not id2text or not queries:
        return []
    
    # For small batches, use simple approach
    if len(queries) < PARALLEL_BATCH_THRESHOLD:
        k = max(1, min(k, len(id2text)))
        q_emb = encode_texts(embed_model, list(queries))
        D, I = index.search(q_emb, k)
        
        results = []
        for row in I:
            ids = []
            for x in row:
                try:
                    doc_id = int(x)
                    if doc_id in id2text:
                        ids.append(doc_id)
                except (ValueError, TypeError):
                    pass
            results.append([id2text[i] for i in ids])
        return results
    
    # For large batches, could implement parallel processing here
    logger.info(f"Processing large batch of {len(queries)} queries")
    return retrieve_batch_top_k_parallel(queries[:PARALLEL_BATCH_THRESHOLD], embed_model, index, id2text, k) + \
           retrieve_batch_top_k_parallel(queries[PARALLEL_BATCH_THRESHOLD:], embed_model, index, id2text, k)

# ------------------------- MAIN ENHANCED -------------------------
def main(user_query: str = "Tell me about DeepSeek R1 efficiency.", 
         screen_path: Optional[str] = None) -> None:
    """Enhanced main with LoRA-optimized RAG system."""
    set_faiss_threads()
    conn = None
    
    try:
        # Initialize database
        conn = init_sqlite(DB_PATH)
        seed_facts(conn)
        facts = get_all_facts(conn)
        
        # Load cached embedding model
        embed_model = get_cached_embed_model(EMBED_MODEL_NAME)
        
        # Legacy corpus
        legacy_docs = [
            "DeepSeek R1 runs efficiently on low VRAM GPUs.",
            "Python is versatile and widely used in AI development.",
            "FAISS helps retrieve relevant embeddings quickly.",
            "RAG integration allows dynamic memory-based responses.",
            "LoRA fine-tuning improves task-specific performance."
        ]

        # Load store and build index with caching
        store = load_store(DOCS_PATH)
        if not store["docs"]:
            # Initialize from legacy docs
            for text in legacy_docs:
                norm = text.strip().lower()
                h = sha1_text(norm)
                if h not in store["hash_to_id"]:
                    store["docs"].append({"id": store["next_id"], "text": text})
                    store["hash_to_id"][h] = store["next_id"]
                    store["next_id"] += 1
        
        # Build index using cached embeddings
        index, doc_ids = build_index_from_embeddings_cached(store, embed_model, conn)
        id2text = id_to_text_map(store)

        # Smart screen capture with deduplication
        store, index = add_screen_memory_smart(store, index, embed_model, conn, screen_path)
        id2text = id_to_text_map(store)

        # Enhanced retrieval
        retrieved_docs = retrieve_top_k_enhanced(user_query, embed_model, index, id2text, TOP_K)
        batch_results = retrieve_batch_top_k_parallel(
            [user_query, "What is FAISS used for?"], embed_model, index, id2text, TOP_K
        )

        # Build and display prompt
        memory_block = "\n".join(facts + retrieved_docs)
        prompt = f"""Use the following memory to answer the user query:
MEMORY:
{memory_block}
USER QUERY:
{user_query}
Answer concisely:"""
        
        print("PROMPT TO LLM:\n", prompt)
        print("\n" + "="*50 + "\n")
        
        # Query LLM with LoRA optimization
        final_answer = query_llm_with_rag(prompt)
        print(f"LLM RESPONSE:\n{final_answer}")
        print("\n" + "="*50)
        
        # Display model info
        model_used = OLLAMA_LORA_MODEL if USE_LORA_MODEL else OLLAMA_BASE_MODEL
        optimization = "LoRA-Enhanced" if USE_LORA_MODEL else "Base Model"
        print(f"\nModel Used: {model_used} ({optimization})")

        # Persist state
        save_store(store, DOCS_PATH)
        
        # Save index with proper error handling
        try:
            def _writer(tmp):
                faiss.write_index(index, tmp)
            atomic_write_bytes(INDEX_TMP, INDEX_PATH, _writer)
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")
        
        logger.info("Successfully completed processing")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if conn:
            try:
                conn.execute("PRAGMA optimize")
            except Exception:
                pass
            conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Tell me about DeepSeek R1 efficiency.")
    parser.add_argument("--screen-path", help="Path to image file for OCR debugging")
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA-optimized model")
    parser.add_argument("--lora-model", default=OLLAMA_LORA_MODEL, help="LoRA model name to use")
    args = parser.parse_args()
    
    # Update LoRA settings from command line
    if args.enable_lora:
        USE_LORA_MODEL = True
        OLLAMA_LORA_MODEL = args.lora_model
    
    main(args.query, args.screen_path)


