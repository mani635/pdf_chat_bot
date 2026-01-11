import streamlit as st
import os
import sqlite3
import hashlib

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------- CONFIG ----------------
st.set_page_config("PDF ChatBot", "📄")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("database.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    pdf_name TEXT,
    vector_path TEXT
)
""")

conn.commit()

# ---------------- HELPERS ----------------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register(username, password):
    try:
        c.execute(
            "INSERT INTO users(username, password) VALUES (?,?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False

def login(username, password):
    c.execute(
        "SELECT id FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    return c.fetchone()

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- AUTH UI ----------------
if st.session_state.user is None:
    st.title("🔐 Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login(u, p)
            if user:
                st.session_state.user = user[0]
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register(u, p):
                st.success("Registered successfully")
            else:
                st.error("Username already exists")

    st.stop()

# ---------------- MAIN APP ----------------
st.title("📄 PDF ChatBot ")

user_id = st.session_state.user
user_vector_dir = f"vectors/user_{user_id}"
os.makedirs(user_vector_dir, exist_ok=True)

# ---------------- LOAD EXISTING VECTORS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None
if os.path.exists(f"{user_vector_dir}/index.faiss"):
    vectorstore = FAISS.load_local(
        user_vector_dir, embeddings, allow_dangerous_deserialization=True
    )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        path = f"{user_vector_dir}/{uploaded_file.name}"

        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        if vectorstore:
            vectorstore.add_documents(chunks)
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)

        vectorstore.save_local(user_vector_dir)

        st.success("PDF permanently added to your account ✅")

# ---------------- CHAT ----------------
llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context.

Context:
{context}

Question:
{question}
""")

query = st.chat_input("Ask from your PDFs")

if query:
    if not vectorstore:
        st.warning("Upload PDF first")
    else:
        retriever = vectorstore.as_retriever()
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        response = chain.invoke(query)
        st.chat_message("assistant").markdown(response.content)

