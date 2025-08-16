# app.py
import os
import requests
import streamlit as st
from typing import List

# ---------- Gemini Config ----------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ---------- Tiny RAG (Knowledge Base) ----------
class TinyRAG:
    def __init__(self):
        self.docs: List[str] = [
            "VPN setup requires defining tunnel interfaces, encryption, and peer IPs.",
            "ACLs should be applied carefully to avoid blocking legitimate traffic.",
            "Routing updates need proper redistribution to prevent loops.",
            "Device commands must match OS syntax (Cisco IOS, Juniper Junos, etc.)."
        ]

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        query_words = query.lower().split()
        scored = []
        for doc in self.docs:
            score = sum(1 for w in query_words if w in doc.lower())
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:k]]

# ---------- Gemini Client ----------
def gemini_generate(api_key: str, prompt: str, max_tokens: int = 300) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
    }
    try:
        resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=30)
        data = resp.json()
    except Exception as e:
        return f"⚠️ Network/Request error: {e}"

    if "candidates" in data and data["candidates"]:
        cand = data["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p and p["text"]:
                return p["text"].strip()
    return "⚠️ Gemini API returned no usable text."

# ---------- Agentic AI Pipeline ----------
def agentic_network_configurator(api_key: str, intent: str, rag: TinyRAG) -> str:
    context = " ".join(rag.retrieve(intent))
    prompt = f"""
You are an expert network engineer AI assistant. Translate the high-level intent into specific, error-free device configurations.

Intent:
{intent}

Knowledge Base:
{context}

Tasks:
1. Generate device commands and configurations.
2. Validate commands syntactically.
3. Suggest optimization if needed.
4. Provide concise explanation for each configuration.

Return a structured report.
"""
    return gemini_generate(api_key, prompt, max_tokens=350)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Intelligent Network Configurator", layout="wide")
st.title("🤖 MANISH - DATA CENTER Intelligent Network Configurator Co-Pilot")

intent_input = st.text_area(
    "Enter high-level network intent (e.g., 'Set up branch office VPN'):",
    height=150
)

if st.button("Generate Configuration"):
    api_key = get_api_key()
    if not api_key:
        st.error("GEMINI_API_KEY not found. Set environment variable or Streamlit secrets.")
    elif not intent_input.strip():
        st.warning("Please enter a network intent.")
    else:
        rag = TinyRAG()
        with st.spinner("Generating configurations..."):
            report = agentic_network_configurator(api_key, intent_input.strip(), rag)
        st.subheader("📝 Configuration Output")
        st.markdown(report)
