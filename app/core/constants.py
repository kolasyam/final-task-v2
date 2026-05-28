"""Centralized constants for the Sales Intelligence System.

All magic values, strings, and numeric constants are defined here
to comply with SonarQube maintainability and reliability rules.

This module should NOT import any application modules to avoid
circular dependencies. It is safe to import from anywhere.
"""

from enum import Enum
from typing import Dict, FrozenSet, List

# =============================================================================
# Application Constants
# =============================================================================

APP_NAME: str = "Sales Intelligence API"
APP_VERSION: str = "2.0.0"
APP_DESCRIPTION: str = (
    "AI-Powered Sales Representative Intelligence Extractor. "
    "Classifies sales notes into issue categories."
)

# =============================================================================
# Category Constants
# =============================================================================

# Raw dataset category values -> normalized output labels
CATEGORY_MAP: Dict[str, str] = {
    "SUPPLY_CHAIN_ISSUE": "supply_chain_delay",
    "RETAILER_RELATIONSHIP_ISSUE": "retailer_dissatisfaction",
    "PRICING_AND_MARGIN_CONFLICT": "pricing_conflict",
    "COMPETITOR_MARKET_PRESSURE": "competitor_pressure",
    "DEMAND_SURGE": "demand_spike",
}

SUPPORTED_CATEGORIES: List[str] = list(CATEGORY_MAP.values())
SUPPORTED_CATEGORIES_SET: FrozenSet[str] = frozenset(SUPPORTED_CATEGORIES)

# Reverse mapping: normalized label -> display name
CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "supply_chain_delay": "Supply Chain Delay",
    "retailer_dissatisfaction": "Retailer Dissatisfaction",
    "pricing_conflict": "Pricing Conflict",
    "competitor_pressure": "Competitor Pressure",
    "demand_spike": "Demand Spike",
}

# =============================================================================
# Keyword Lookup Tables (used by OllamaClient.extract_category)
# =============================================================================

CATEGORY_KEYWORDS: Dict[str, FrozenSet[str]] = {
    "supply_chain_delay": frozenset({
        "supply", "chain", "delay", "stock", "shortage", "delivery",
        "replenish", "shipment", "backlog", "warehouse", "inventory",
        "mismatch", "running", "out", "fast", "movers", "movement",
    }),
    "retailer_dissatisfaction": frozenset({
        "retailer", "dissatisfaction", "unhappy", "complain", "angry",
        "frustrated", "poor", "service", "bad", "experience", "relationship",
        "issue", "dissatisfied",
    }),
    "pricing_conflict": frozenset({
        "pricing", "conflict", "price", "dispute", "margin", "discount",
        "expensive", "cheap", "billing", "charge", "rate", "cost",
    }),
    "competitor_pressure": frozenset({
        "competitor", "pressure", "competition", "rival", "alternative",
        "switching", "market", "share", "launched", "campaign",
    }),
    "demand_spike": frozenset({
        "demand", "spike", "surge", "overflow", "high", "volume",
        "rush", "stockout", "unexpected", "high", "demand",
    }),
}


class InferenceMethod(str, Enum):
    """Inference method identifiers returned in prediction responses."""

    OLLAMA_PROMPT_MODEL = "ollama_prompt_model"
    OLLAMA_BASE = "ollama_base"
    OLLAMA_FINETUNED = "ollama_finetuned"
    SKLEARN_TFIDF = "sklearn_tfidf"
    QLORA_DIRECT = "qlora_direct"


# =============================================================================
# Ollama Constants
# =============================================================================

PROMPT_MODEL_NAME: str = "gemma-sales-intel"
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_MODEL_NAME: str = "gemma:2b"
DEFAULT_OLLAMA_TIMEOUT: int = 120
OLLAMA_MAX_TOKENS: int = 30
OLLAMA_TEMPERATURE: float = 0.1
OLLAMA_TOP_P: float = 0.9

CLASSIFICATION_PROMPT: str = (
    "You are a sales intelligence analyst. Classify the following "
    "sales representative field note into exactly ONE of these categories:\n\n"
    "1. supply_chain_delay — stock shortages, delivery delays, replenishment issues\n"
    "2. retailer_dissatisfaction — complaints, unhappy retailers, service issues\n"
    "3. pricing_conflict — price disputes, margin concerns, discount conflicts\n"
    "4. competitor_pressure — competitor actions, market share threats, rival offers\n"
    "5. demand_spike — unexpected demand surges, stockout from high volume\n\n"
    "Sales Note: {note}\n\n"
    "Respond with ONLY the category name (e.g., supply_chain_delay). "
    "No explanation, no extra text.\n"
    "Category:"
)

CATEGORY_PREFIXES_TO_STRIP: List[str] = [
    "issue category:",
    "category:",
    "the issue category is",
    "the category is",
    "-",
]

# =============================================================================
# Circuit Breaker Constants
# =============================================================================

DEFAULT_CIRCUIT_BREAKER_THRESHOLD: int = 5
DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS: float = 30.0

# =============================================================================
# Retry Constants
# =============================================================================

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_BASE_DELAY: float = 1.0
DEFAULT_RETRY_MAX_DELAY: float = 10.0
DEFAULT_RETRY_BACKOFF_FACTOR: float = 2.0

# =============================================================================
# Security Constants
# =============================================================================

API_KEY_HEADER: str = "X-API-Key"
CORRELATION_ID_HEADER: str = "X-Correlation-ID"

AUTH_EXEMPT_PATHS: List[str] = [
    "/api/v1/health",
    "/api/v1/categories",
    "/docs",
    "/openapi.json",
    "/redoc",
]

# =============================================================================
# Rate Limiting Constants
# =============================================================================

DEFAULT_RATE_LIMIT_RPM: int = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: int = 60

# =============================================================================
# API Constants
# =============================================================================

DEFAULT_API_HOST: str = "0.0.0.0"
DEFAULT_API_PORT: int = 8000
DEFAULT_STREAMLIT_PORT: int = 8501
DEFAULT_CORS_ORIGINS: str = "http://localhost:8501,http://localhost:3000"

# =============================================================================
# Model / Training Constants
# =============================================================================

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_N_ESTIMATORS: int = 200
DEFAULT_MAX_FEATURES: int = 5000

# =============================================================================
# Storage Constants
# =============================================================================

DEFAULT_CSV_PATH: str = "data/predictions.csv"
DEFAULT_JSONL_PATH: str = "data/predictions.jsonl"

# =============================================================================
# Input Validation Constants
# =============================================================================

MIN_NOTE_LENGTH: int = 1
MAX_NOTE_LENGTH: int = 1000
MAX_BATCH_SIZE: int = 50

# =============================================================================
# Text Abbreviation Map (used by TextPreprocessor)
# =============================================================================

ABBREVIATION_MAP: Dict[str, str] = {
    "stk": "stock",
    "nt": "not",
    "cmng": "coming",
    "frm": "from",
    "dys": "days",
    "wks": "weeks",
    "dlvry": "delivery",
    "dlvr": "deliver",
    "repl": "replenishment",
    "ret": "retailer",
    "reps": "representative",
    "repr": "representative",
    "cust": "customer",
    "custs": "customers",
    "pr": "price",
    "prc": "price",
    "prblm": "problem",
    "prblms": "problems",
    "cmpln": "complain",
    "cmplns": "complains",
    "cmplng": "complaining",
    "inc": "increase",
    "dcr": "decrease",
    "dmn": "demand",
    "spk": "spike",
    "cmptr": "competitor",
    "cmptrs": "competitors",
    "invntry": "inventory",
    "invtry": "inventory",
    "wrg": "wrong",
    "ordr": "order",
    "ordrs": "orders",
    "amt": "amount",
    "qtty": "quantity",
    "qty": "quantity",
    "pd": "paid",
    "pdbl": "payable",
    "bl": "bill",
    "bllng": "billing",
    "pdct": "product",
    "pdcts": "products",
    "dlay": "delay",
    "dlays": "delays",
    "dlayed": "delayed",
    "ot": "out of",
    "stkout": "stockout",
    "bck": "back",
    "bk": "back",
    "rgnl": "regional",
    "reg": "region",
    "dlv": "deliver",
    "dlvd": "delivered",
    "shrtg": "shortage",
    "srvc": "service",
    "srvcng": "servicing",
    "ups": "upsell",
    "crs": "cross",
    "sls": "sales",
    "slsmn": "salesman",
    "slsprsn": "salesperson",
    "mkt": "market",
    "mktng": "marketing",
    "ftr": "feature",
    "bnft": "benefit",
    "whsl": "wholesale",
    "rtlr": "retailer",
    "rtl": "retail",
    "mgn": "margin",
    "mrgn": "margin",
    "ntwk": "network",
    "dl": "deal",
    "dls": "deals",
    "cntrct": "contract",
    "cntrcts": "contracts",
    "rqrmnt": "requirement",
    "qlyty": "quality",
    "qlty": "quality",
    "std": "standard",
    "hvy": "heavy",
    "lgt": "light",
    "lrg": "large",
    "sml": "small",
    "mk": "make",
    "sv": "save",
    "gv": "give",
    "gt": "get",
    "gl": "goal",
    "trg": "target",
    "trgt": "target",
    "incntv": "incentive",
    "rt": "rate",
    "rtng": "rating",
    "rvw": "review",
    "pndng": "pending",
    "apprv": "approve",
    "rjct": "reject",
    "rjctd": "rejected",
    "apprvl": "approval",
    "tx": "tax",
    "wh": "warehouse",
    "whs": "warehouse",
    "cncl": "cancel",
    "cnfld": "confidential",
    "pndt": "pending",
    "rfr": "refer",
    "rmnd": "remind",
    "rmndr": "reminder",
    "shpng": "shipping",
    "shp": "ship",
    "rcv": "receive",
    "rcvd": "received",
    "snd": "send",
    "sndng": "sending",
    "hdqtr": "headquarters",
    "offc": "office",
    "dept": "department",
    "div": "division",
    "mgr": "manager",
    "hod": "head",
}
