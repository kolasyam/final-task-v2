"""Text preprocessing service for sales representative notes."""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocesses sales notes for model input."""

    # Common sales field abbreviations mapped to full forms
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

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        logger.info("TextPreprocessor initialized")

    def to_lower(self, text: str) -> str:
        """Convert text to lowercase.

        Args:
            text: Input text.

        Returns:
            Lowercased text.
        """
        return text.lower()

    def clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace and strip.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        return re.sub(r"\s+", " ", text).strip()

    def clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation while preserving meaningful characters.

        Args:
            text: Input text.

        Returns:
            Text with cleaned punctuation.
        """
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[,]{2,}", ",", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
        return text

    def normalize_typos(self, text: str) -> str:
        """Normalize common typos and abbreviations.

        Args:
            text: Input text.

        Returns:
            Text with expanded abbreviations.
        """
        words: List[str] = text.split()
        normalized: List[str] = []
        for word in words:
            cleaned_word: str = re.sub(r"[^a-zA-Z]", "", word.lower())
            if cleaned_word in self.ABBREVIATION_MAP:
                normalized.append(self.ABBREVIATION_MAP[cleaned_word])
            else:
                normalized.append(word)
        return " ".join(normalized)

    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline to text.

        Args:
            text: Raw input text from sales representative note.

        Returns:
            Cleaned and normalized text ready for model input.

        Raises:
            ValueError: If input text is empty or whitespace.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        text = self.to_lower(text)
        text = self.clean_punctuation(text)
        text = self.normalize_typos(text)
        text = self.clean_whitespace(text)

        logger.debug("Preprocessed text: %s", text)
        return text
