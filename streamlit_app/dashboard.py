"""Streamlit dashboard for the Sales Intelligence Extractor.

Provides a professional web interface for classifying sales
representative notes into issue categories. Communicates with
the Ollama (prompt-engineered gemma-sales-intel) + scikit-learn backend.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure parent directory is in path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Sales Intelligence Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Helper Functions ---

def check_api_health() -> Optional[Dict[str, Any]]:
    """Check if the FastAPI backend is reachable.

    Returns:
        Health response dictionary or None if unreachable.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def predict_category(note: str) -> Optional[Dict[str, Any]]:
    """Call the FastAPI prediction endpoint.

    Args:
        note: Sales representative note text.

    Returns:
        Prediction response dictionary or None if request fails.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict",
            json={"rep_note": note},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"API Error: {exc}")
        return None


def get_categories() -> List[str]:
    """Retrieve supported categories from the API.

    Returns:
        List of category strings.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/categories", timeout=5,
        )
        response.raise_for_status()
        return response.json().get("categories", [])
    except requests.exceptions.RequestException:
        return [
            "supply_chain_delay",
            "retailer_dissatisfaction",
            "pricing_conflict",
            "competitor_pressure",
            "demand_spike",
        ]


# --- Page Layout ---

st.title("📊 AI-Powered Sales Representative Intelligence Extractor")
st.markdown("Classify field notes into issue categories using a fine-tuned AI model")
st.markdown("---")

# --- Sidebar ---

st.sidebar.header("🔌 System Status")
health: Optional[Dict[str, Any]] = check_api_health()

if health:
    st.sidebar.success("✅ API Connected")
    ollama_status = "✅ Available" if health.get("ollama_available") else "❌ Unavailable"
    sklearn_status = "✅ Loaded" if health.get("sklearn_available") else "❌ Not Loaded"
    model_name = health.get("model_name", "N/A")

    # Show model status prominently
    if "gemma-sales-intel" in model_name:
        st.sidebar.markdown(f"🧠 **Model:** `{model_name}`")
        st.sidebar.success("🎯 **Prompt-engineered model active** — optimized for your dataset")
    elif health.get("ollama_available"):
        st.sidebar.markdown(f"🤖 **Model:** `{model_name}`")
        st.sidebar.warning("⚡ Using base model — run prompt engineering for best accuracy")
    else:
        st.sidebar.markdown(f"⚡ **Mode:** sklearn fallback")

    st.sidebar.markdown(f"**Ollama:** {ollama_status}")
    st.sidebar.markdown(f"**Sklearn Fallback:** {sklearn_status}")

    # Prompt engineering instructions sidebar
    if "gemma-sales-intel" not in model_name:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Enable Prompt Engineering")
        st.sidebar.code(
            "python -m training.generate_modelfile\n"
            "ollama create gemma-sales-intel "
            "-f training/saved_model/Modelfile",
            language="bash",
        )
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.info("Start the API: `python -m app.main`")

st.markdown("---")
st.sidebar.header("📋 Supported Categories")
categories: List[str] = get_categories()
category_labels: Dict[str, str] = {
    "supply_chain_delay": "⛓️ Supply Chain Delay",
    "retailer_dissatisfaction": "😟 Retailer Dissatisfaction",
    "pricing_conflict": "💰 Pricing Conflict",
    "competitor_pressure": "🏢 Competitor Pressure",
    "demand_spike": "📈 Demand Spike",
}
for cat in categories:
    label = category_labels.get(cat, cat)
    st.sidebar.markdown(f"- **{label}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 API Endpoint")
st.sidebar.code(f"{API_BASE_URL}/api/v1/predict", language="text")

# --- Main Content: Prediction Section ---

st.header("🔍 Issue Category Prediction")

col1, col2 = st.columns([3, 2])

with col1:
    note_input: str = st.text_area(
        "Enter Sales Representative Note:",
        height=150,
        placeholder=(
            "e.g., Retailer reported stock running out of fast movers, "
            "customers already asking questions about product availability."
        ),
        help="Enter a field note from a sales representative. "
        "The AI will classify the issue category.",
        key="note_input",
        max_chars=1000,
    )

    char_count = len(note_input) if note_input else 0
    st.caption(f"Characters: {char_count}/1000")

    predict_btn: bool = st.button(
        "🔮 Predict Category",
        type="primary",
        disabled=not note_input or not note_input.strip(),
        use_container_width=True,
    )

with col2:
    st.markdown("### 💡 Sample Notes")
    examples: List[Dict[str, str]] = [
        {
            "label": "⛓️ Supply Chain",
            "text": (
                "Store reported running out of fast movers, "
                "customers already asking questions."
            ),
        },
        {
            "label": "😟 Retailer Issue",
            "text": (
                "Store manager unhappy with service quality, "
                "threatened to reduce orders."
            ),
        },
        {
            "label": "💰 Pricing",
            "text": (
                "Retailer says our prices are too high compared "
                "to competitors, wants discount."
            ),
        },
        {
            "label": "🏢 Competitor",
            "text": (
                "Competitor launched aggressive pricing campaign "
                "in the region, retailers switching."
            ),
        },
        {
            "label": "📈 Demand Spike",
            "text": (
                "Unexpected surge in demand, stockout situation "
                "reported across multiple stores."
            ),
        },
    ]
    for example in examples:
        if st.button(example["label"], key=f"example_{example['label']}"):
            st.session_state["note_input"] = example["text"]
            st.rerun()

# --- Prediction Result ---

if predict_btn and note_input and note_input.strip():
    with st.spinner("Analyzing note with AI..."):
        start_time = time.time()
        result: Optional[Dict[str, Any]] = predict_category(note_input)
        total_latency = time.time() - start_time

    if result:
        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            category_display = result["issue_category"].replace("_", " ").title()
            st.metric("Issue Category", category_display)

        with result_col2:
            confidence_pct = f"{result['confidence']:.1%}"
            st.metric("Confidence", confidence_pct)

        with result_col3:
            st.metric("Latency", f"{total_latency:.2f}s")

        # Method badge
        method = result.get("method", "unknown")
        if method == "ollama_prompt_model":
            st.success("🧠 Inference: Prompt-engineered gemma-sales-intel (optimized for your data)")
        elif method == "ollama_base":
            st.info("🤖 Inference: Base gemma:2b (zero-shot)")
        else:
            st.info("⚡ Inference: scikit-learn TF-IDF + RandomForest")

        # Reasoning
        with st.expander("📝 Reasoning", expanded=True):
            st.write(result.get("reasoning", "No reasoning available."))

        # Raw response
        with st.expander("🔧 Raw API Response"):
            st.json(result)

# --- Analytics Section ---

st.markdown("---")
st.header("📈 Prediction Analytics")

analytics_col1, analytics_col2 = st.columns(2)

with analytics_col1:
    st.subheader("Category Distribution")
    predictions_csv: str = "data/predictions.csv"
    if os.path.exists(predictions_csv):
        try:
            df: pd.DataFrame = pd.read_csv(predictions_csv)
            if not df.empty and "issue_category" in df.columns:
                counts: pd.Series = df["issue_category"].value_counts()
                chart_data = pd.DataFrame({
                    "Category": [
                        cat.replace("_", " ").title() for cat in counts.index
                    ],
                    "Count": counts.values,
                })
                st.bar_chart(chart_data.set_index("Category"))
            else:
                st.info("No predictions recorded yet.")
        except Exception as exc:
            st.warning(f"Could not load predictions: {exc}")
    else:
        st.info("No predictions file found. Make some predictions first!")

with analytics_col2:
    st.subheader("Recent Predictions")
    if os.path.exists(predictions_csv):
        try:
            history_df: pd.DataFrame = pd.read_csv(predictions_csv)
            if not history_df.empty:
                display_cols: List[str] = [
                    col for col in ["timestamp", "issue_category"]
                    if col in history_df.columns
                ]
                if display_cols:
                    st.dataframe(
                        history_df[display_cols].tail(10),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        history_df.tail(10),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("No prediction history yet.")
        except Exception as exc:
            st.warning(f"Could not load history: {exc}")
    else:
        st.info("No predictions file found.")

# --- Footer ---

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Sales Intelligence Extractor v1.0 | "
    "Fine-tuned Ollama gemma:2b + scikit-learn | "
    "Built with FastAPI + Streamlit | "
    "Enterprise-Grade AI Infrastructure</div>",
    unsafe_allow_html=True,
)
