import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Mini-RAG Demo", layout="wide")

st.title("📚 Mini RAG Document QA System")

# ---- Upload section ----
st.subheader("1️⃣ Ingest a PDF document")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
lang = st.text_input("Language hint (optional)", "")

if uploaded:
    if st.button("Ingest"):
        with st.spinner("Uploading & indexing..."):
            files = {"file": (uploaded.name, uploaded.read(), "application/pdf")}
            data = {"language": lang}
            r = requests.post(f"{API_URL}/ingest", files=files, data=data)
            if r.ok:
                st.success(f"✅ {r.json()['chunks']} chunks indexed.")
            else:
                st.error(r.text)

st.markdown("---")

# ---- Query section ----
st.subheader("2️⃣ Ask questions")
query = st.text_area("Enter your question here")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        payload = {"query": query}
        r = requests.post(f"{API_URL}/query", json=payload)
        if r.ok:
            data = r.json()
            st.markdown("### 🧠 Answer")
            st.write(data["answer"])
            st.markdown("**Provenance:**")
            for p in data.get("provenance", []):
                st.write(f"📄 Doc: {p.get('doc_id')}  |  Page: {p.get('page')}")
            st.caption(f"{data.get('latency_ms', 0)} ms")
        else:
            st.error(r.text)
