import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load env
load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Page config
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📄",
    layout="wide"
)

# SIDEBAR
st.sidebar.title("💬 Chat History")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

for q, a in reversed(st.session_state.chat_history):
    st.sidebar.markdown(f"**🧑 {q}**")
    st.sidebar.markdown(f"🤖 {a}")
    st.sidebar.markdown("---")

#  MAIN UI 
st.markdown("<h1 style='font-size:44px; text-align:center;'>📄 Document AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size:22px; text-align:center;'>Upload a document and chat with it</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload your document (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"]
)

#  FILE LOADER 
def load_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    return ""

#  CHUNKING 
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

#  EMBEDDINGS 
def create_embeddings(chunks):
    return embedding_model.encode(chunks)

#  FAISS 
def build_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

#  SEARCH 
def search_chunks(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

#  LLM 
def get_answer(question, context):
    prompt = f"""
You are a SMART document-based AI assistant.

You must behave in 2 modes:

========================
1. FACT MODE (STRICT)
========================
Use this when question is:
- factual (skills, education, experience, tools)
RULES:
- Answer ONLY from context
- If not present: say "Not mentioned in the document"
- No assumptions

========================
2. SMART MODE (INFERENCE)
========================
Use this when question is:
- subjective (e.g., "Is candidate good?", "How skilled?", "Is this strong?")
RULES:
- Do NOT hallucinate
- Infer ONLY based on evidence in context
- Be cautious and balanced
- Use phrases like:
  "Based on the available information..."
  "The candidate shows strong indicators such as..."

========================
OUTPUT STYLE
========================
- Clear
- Professional
- Human-like
- No repetition of raw text

========================
CONTEXT
========================
{context}

QUESTION
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
#  MAIN APP 
if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    text = load_file(uploaded_file)
    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    # Chat input (clean + always available)
    question = st.chat_input("Ask anything about your document...")

    if question:
        results = search_chunks(question, index, chunks)
        context = "\n\n".join(results)

        answer = get_answer(question, context)

        st.session_state.chat_history.append((question, answer))

    #  CHAT DISPLAY (CLEAN UI) 
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            st.markdown(a)