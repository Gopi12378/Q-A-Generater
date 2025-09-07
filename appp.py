import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from typing import List, Dict

from PyPDF2 import PdfReader
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Embeddings / vector DB
from sentence_transformers import SentenceTransformer
import numpy as np

# Optional: chromadb for persistent vector store
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False

# --------------------------
# Config / API
# --------------------------
from dotenv import load_dotenv


load_dotenv()  # load variables from .env

api_key = os.getenv("GROQ_API_KEY")

GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Chroma persistence directory (optional)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "question_papers")

# thresholds
SIMILARITY_THRESHOLD = 0.20

# --------------------------
# Globals for RAG / Vector DB
# --------------------------
embed_model = None
chroma_client = None
chroma_collection = None
text_chunks: List[str] = []

# --------------------------
# Helpers
# --------------------------
def safe_rerun():
    try:
        st.session_state._rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# --------------------------
# Groq API call
# --------------------------
def call_groq(prompt: str, api_key: str = api_key, model: str = DEFAULT_MODEL, temperature: float = 0.35) -> str:
    if not api_key:
        return "⚠️ No API key found. Please set GROQ_API_KEY in .env."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful exam assistant that generates exam-style questions and concise answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    r = requests.post(GROQ_API, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# --------------------------
# PDF utils
# --------------------------
def extract_text_from_pdf(fileobj) -> str:
    reader = PdfReader(fileobj)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def extract_question_like_lines(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    qns = [l for l in lines if re.search(r'\bQ[0-9]+\.|^[0-9]+\s*[).:-]|[?]$', l)]
    if not qns:
        qns = [l for l in lines if len(l) > 30][:200]
    return qns

# --------------------------
# Export helpers
# --------------------------
def make_docx(text: str) -> BytesIO:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

def make_pdf(text: str) -> BytesIO:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in text.splitlines():
        c.drawString(40, y, line[:200])
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    buf.seek(0)
    return buf

# --------------------------
# Embedding helpers
# --------------------------
def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embed_model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return np.dot(An, Bn.T)

def max_similarity_to_corpus_embeddings(item: str, corpus_texts: List[str], corpus_embeddings: np.ndarray) -> float:
    if not corpus_texts:
        return 0.0
    q_emb = embed_texts([item])
    sims = cosine_sim_matrix(q_emb, corpus_embeddings).ravel()
    return float(sims.max()) if sims.size else 0.0

def compute_confidence_metrics(generated_questions: List[str], uploaded_corpus: List[str]) -> Dict:
    if not generated_questions:
        return {"confidences": [], "avg_confidence": 0.0, "low_count": 0, "low_pct": 100.0}
    if uploaded_corpus:
        corpus_embeddings = embed_texts(uploaded_corpus)
        confidences = [max_similarity_to_corpus_embeddings(q, uploaded_corpus, corpus_embeddings) for q in generated_questions]
    else:
        confidences = [0.0 for _ in generated_questions]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_count = sum(1 for c in confidences if c < SIMILARITY_THRESHOLD)
    low_pct = low_count / len(confidences) * 100
    return {"confidences": confidences, "avg_confidence": avg_conf, "low_count": low_count, "low_pct": low_pct}

# --------------------------
# Chroma DB helpers
# --------------------------
def get_chroma_client(persist_directory: str = CHROMA_PERSIST_DIR):
    global chroma_client, chroma_collection
    if not CHROMADB_AVAILABLE:
        return None, None
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    if chroma_collection is None:
        chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return chroma_client, chroma_collection

def clear_chroma_collection(collection):
    try:
        collection.delete(where={})
    except Exception:
        try:
            existing = collection.get()
            ids = existing.get('ids', [])
            if ids:
                collection.delete(ids=ids)
        except Exception:
            pass

def build_vector_index(texts: List[str], chunk_size=300):
    global text_chunks
    text_chunks = []
    for txt in texts:
        words = txt.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                text_chunks.append(chunk)
    if not text_chunks:
        return
    embeddings = embed_texts(text_chunks)
    client, collection = get_chroma_client()
    if client and collection:
        clear_chroma_collection(collection)
        ids = [f"doc_{i}" for i in range(len(text_chunks))]
        metadatas = [{"source": "uploaded", "chunk_i": i} for i in range(len(text_chunks))]
        try:
            collection.add(ids=ids, documents=text_chunks, metadatas=metadatas, embeddings=embeddings.tolist())
        except Exception:
            collection.add(ids=ids, documents=text_chunks, metadatas=metadatas, embeddings=embeddings)

def retrieve_context(query: str, top_k=3) -> str:
    global text_chunks
    client, collection = get_chroma_client()
    if client and collection:
        try:
            res = collection.query(query_texts=[query], n_results=top_k, include=['documents', 'distances'])
            docs = res.get('documents', [[]])[0]
            return "\n\n".join([d for d in docs if d])
        except Exception:
            pass
    if not text_chunks:
        return ""
    q_emb = embed_texts([query])
    chunk_embs = embed_texts(text_chunks)
    sims = cosine_sim_matrix(q_emb, chunk_embs).ravel()
    top_idx = sims.argsort()[::-1][:top_k]
    retrieved = [text_chunks[i] for i in top_idx if i < len(text_chunks)]
    return "\n\n".join(retrieved)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="📚 Question Paper Assistant", layout="wide")
st.markdown('<div class="big-title">📚 Question Paper Assistant — Clean & Professional (Embeddings + Chroma)</div>', unsafe_allow_html=True)

# Initialize session state
for key, default in {
    "page": 1,
    "uploaded_texts": [],
    "uploaded_names": [],
    "uploaded_questions": [],
    "selected_subjects": [],
    "template": None,
    "papers": [],
    "current_paper": 0,
    "top_keywords": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------
# Step 1: Upload PDFs
# --------------------------
if st.session_state["page"] == 1:
    st.markdown("### Step 1 — Select Subjects & Upload Past Papers")
    subj = st.text_input("Enter Subjects (comma-separated)", value="Chemistry")
    uploaded = st.file_uploader("Upload Past Papers (PDF)", type=["pdf"], accept_multiple_files=True)

    if subj:
        st.session_state["selected_subjects"] = [s.strip() for s in subj.split(",") if s.strip()]

    if uploaded:
        st.session_state["uploaded_texts"].clear()
        st.session_state["uploaded_names"].clear()
        st.session_state["uploaded_questions"].clear()
        for f in uploaded:
            txt = extract_text_from_pdf(f)
            st.session_state["uploaded_texts"].append(txt)
            st.session_state["uploaded_names"].append(f.name)
            st.session_state["uploaded_questions"].extend(extract_question_like_lines(txt))

        # Extract top keywords
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect = TfidfVectorizer(stop_words="english", max_features=2000)
            X = vect.fit_transform(st.session_state["uploaded_questions"])
            scores = X.sum(axis=0).A1
            terms = vect.get_feature_names_out()
            st.session_state["top_keywords"] = [t for t, _ in sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:30]]
        except Exception:
            st.session_state["top_keywords"] = []

        build_vector_index(st.session_state["uploaded_texts"])
        st.success(f"Uploaded {len(uploaded)} file(s). Extracted ~{len(st.session_state['uploaded_questions'])} question lines.")

    cols = st.columns([1, 1, 1])
    if cols[2].button("Next ►"):
        st.session_state["page"] = 2
        safe_rerun()

# --------------------------
# Remaining Steps (2-4) are similar with same safe_rerun handling and embeddings integration
# --------------------------
# Due to space, the rest (template selection, paper generation, confidence check, topic Q&A) can follow the same logic as in your previous script.
# --------------------------


# --------------------------
# Step 2 — Template selection & preview
# --------------------------
elif st.session_state["page"] == 2:
    st.markdown("### Step 2 — Choose Template & Preview Structure")
    if not st.session_state["uploaded_texts"]:
        st.warning("Please upload PDFs in Step 1 first.")
        if st.button("Back ◄"):
            st.session_state["page"] = 1
            safe_rerun()
    else:
        files = st.session_state["uploaded_names"]
        sel_idx = st.selectbox("Select Template File", options=list(range(len(files))), format_func=lambda i: files[i])
        st.session_state["template"] = sel_idx

        left, right = st.columns([3, 1.25])
        with left:
            st.text_area("Template preview (first 4000 chars)", value=st.session_state["uploaded_texts"][sel_idx][:4000], height=400)
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Template Details**")
            st.write(f"- File: **{files[sel_idx]}**")
            sample_txt = st.session_state["uploaded_texts"][sel_idx]
            qcount = len(extract_question_like_lines(sample_txt))
            st.write(f"- Detected question-like lines: **{qcount}**")
            kw_display = ", ".join(st.session_state.get("top_keywords", [])[:12])
            if kw_display:
                st.write("- Top keywords:")
                st.write(kw_display)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>')
        cols = st.columns([1, 1, 1])
        if cols[0].button("Back ◄"):
            st.session_state["page"] = 1
            safe_rerun()
        if cols[2].button("Next ►"):
            st.session_state["page"] = 3
            safe_rerun()

# --------------------------
# Step 3 — Generate Papers with RAG context injection
# --------------------------
elif st.session_state["page"] == 3:
    st.markdown("### Step 3 — Generate Question Papers")
    if not st.session_state["uploaded_texts"]:
        st.warning("Please upload PDFs in Step 1 first.")
        if st.button("Back ◄"):
            st.session_state["page"] = 2
            safe_rerun()
    else:
        subj_list = st.session_state.get("selected_subjects", [])
        st.markdown(f"**Subjects:** {', '.join(subj_list) or '—'}")
        q_count = st.number_input("Number of questions per paper", min_value=1, max_value=200, value=10)
        num_papers = st.number_input("Number of papers to generate", min_value=1, max_value=5, value=1)
        temp = st.slider("Generation temperature (creativity level)", 0.0, 1.0, 0.25)

        st.markdown('<div class="spacer"></div>')
        cols = st.columns([1, 1, 1])
        if cols[0].button("Back ◄"):
            st.session_state["page"] = 2
            safe_rerun()

        if cols[2].button("Generate ►"):
            template_idx = st.session_state.get("template", 0)
            template_text = st.session_state["uploaded_texts"][template_idx]
            top_kws = st.session_state.get("top_keywords", [])[:40]
            kw_text = ", ".join(top_kws)

            st.session_state["papers"].clear()
            progress_placeholder = st.empty()

            # Build query string for retrieval from subjects and top keywords
            retrieval_query = ", ".join(subj_list + top_kws)
            retrieved_context = retrieve_context(retrieval_query, top_k=3)

            for p in range(int(num_papers)):
                prompt = f"""
You are an exam paper generator. Follow these rules strictly:
Use the following context from uploaded materials to guide generation:
{retrieved_context}

1) Preserve the template structure exactly (headings, numbering).
2) Replace questions with concise exam-style questions relevant to: {', '.join(subj_list)}.
3) Use only these keywords/topics: {kw_text}.
4) Fill up to {q_count} question slots.
Output the full paper in plain text.
--- TEMPLATE START ---
{template_text[:3000]}
--- TEMPLATE END ---
"""
                try:
                    out = call_groq(prompt, api_key=api_key, model=DEFAULT_MODEL, temperature=float(temp))
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    out = ""
                st.session_state["papers"].append(out)
                progress_placeholder.text(f"Generating paper {p+1}/{int(num_papers)}...")
            progress_placeholder.empty()
            st.success(f"Generated {len(st.session_state['papers'])} paper(s).")
            st.session_state["current_paper"] = 0
            st.session_state["page"] = 4
            safe_rerun()

# --------------------------
# Step 4 — View Papers & Confidence Score with Topic Q&A
# --------------------------
elif st.session_state["page"] == 4:
    st.markdown("### Step 4 — View Generated Papers & Confidence Scores")

    if not st.session_state["papers"]:
        st.warning("No generated papers available. Please generate papers in Step 3.")
        if st.button("Back ◄"):
            st.session_state["page"] = 3
            safe_rerun()
    else:
        idx = st.session_state.get("current_paper", 0)
        paper_text = st.session_state["papers"][idx] or "(empty output)"

        left, right = st.columns([3, 1.3])

        with left:
            st.subheader(f"📄 Paper {idx+1} of {len(st.session_state['papers'])}")
            st.text_area("Generated Paper Preview", value=paper_text, height=450)
            st.download_button(
                "📥 Download as .docx",
                make_docx(paper_text),
                file_name=f"paper_{idx+1}.docx",
                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
            st.download_button(
                "📥 Download as .pdf",
                make_pdf(paper_text),
                file_name=f"paper_{idx+1}.pdf",
                mime='application/pdf'
            )

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Confidence Score Analysis**")
            gen_qs = extract_question_like_lines(paper_text)
            uploaded_corpus = st.session_state.get("uploaded_questions", [])
            conf = compute_confidence_metrics(gen_qs, uploaded_corpus)
            avgc = conf["avg_confidence"]
            st.metric(label="Average Confidence", value=f"{avgc:.3f}", delta=None)
            st.write(f"Low-confidence Questions (<{SIMILARITY_THRESHOLD}): {conf['low_count']} ({conf['low_pct']:.1f}%)")
            st.markdown("</div>", unsafe_allow_html=True)

            nav_left, nav_right = st.columns(2)
            with nav_left:
                if st.button("◄ Previous Paper") and st.session_state["current_paper"] > 0:
                    st.session_state["current_paper"] -= 1
                    safe_rerun()
            with nav_right:
                if st.button("Next Paper ►") and st.session_state["current_paper"] < len(st.session_state["papers"]) - 1:
                    st.session_state["current_paper"] += 1
                    safe_rerun()

        st.markdown("---")

        with st.expander("Per-question Confidence Details"):
            if not gen_qs:
                st.write("No question-like lines detected in generated paper.")
            else:
                for i, q in enumerate(gen_qs, 1):
                    c = conf["confidences"][i-1] if i-1 < len(conf["confidences"]) else 0.0
                    badge = "✅" if c >= SIMILARITY_THRESHOLD else "⚠️"
                    st.markdown(f"**{badge} Q{i}**  \n{q}")

        st.markdown("---")

        st.subheader("📚 Topic Q&A Generator & Confidence Check")
        topic = st.text_input("Enter topic for Q&A", key="topic_input", placeholder="E.g. chemical reactions, machine learning")
        n_qa = st.slider("Number of Q&A pairs to generate", 1, 10, 5, key="topic_qcount")

        if st.button("Generate Topic Q&A"):
            if not topic.strip():
                st.warning("Please enter a topic to generate Q&A.")
            else:
                qa_prompt = f"""
Generate {n_qa} exam-style questions about: {topic}.
For each question, output Qn. ... then Ans: ... (concise 2-3 lines).
Format exactly.
"""
                try:
                    qa_out = call_groq(qa_prompt, api_key=api_key, model=DEFAULT_MODEL, temperature=0.25)
                except Exception as e:
                    st.error(f"Topic Q&A generation failed: {e}")
                    qa_out = ""

                st.text_area("Generated Topic Q&A", value=qa_out, height=320, key="qa_output_area")

                qa_lines = [l for l in qa_out.splitlines() if l.strip() and re.match(r'^Q\d+\.', l.strip(), flags=re.IGNORECASE)]
                if qa_lines and st.session_state.get("uploaded_questions"):
                    # compute confidences with embeddings
                    corpus_qs = st.session_state.get("uploaded_questions")
                    corpus_embeddings = embed_texts(corpus_qs)
                    qa_confs = [max_similarity_to_corpus_embeddings(q, corpus_qs, corpus_embeddings) for q in qa_lines]
                    avg_qa = sum(qa_confs) / len(qa_confs)
                    st.metric("Topic Q Average Confidence", f"{avg_qa:.3f}")
                    # per-question badges suppressed per user request
                    if avg_qa < 0.25:
                        st.error("Topic Q&A appears low-confidence compared to uploaded materials.")
                    else:
                        st.success("Topic Q&A aligns well with uploaded materials.")
                else:
                    st.info("Not enough uploaded material available to compute confidence for Topic Q&A.")



# --------------------------
# End of UI
# --------------------------

# Usage notes (in-code):
# 1) Install dependencies:
#    pip install streamlit python-dotenv requests pypdf2 python-docx reportlab sentence-transformers chromadb
# 2) Set env vars: GROQ_API_KEY and optionally CHROMA_PERSIST_DIR
# 3) Run: streamlit run question_paper_assistant_vector_chroma.py
