import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# --- API Key (Replace with your own) ---
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Global Variables ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def chunk_and_store_text(text):
    """Splits text and stores embeddings in FAISS."""
    global index, chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    embeddings = np.array(embedding_model.encode(chunks))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

def retrieve_relevant_text(query, top_k=3):
    """Retrieves top-K relevant text chunks for a given query."""
    if index is None or not chunks:
        return "No document uploaded yet."

    query_embedding = np.array(embedding_model.encode([query]))
    _, indices = index.search(query_embedding, top_k)
    return "\n".join([chunks[i] for i in indices[0]])

def call_openrouter(prompt, context=""):
    """Calls OpenRouter's API with contextual information for in-depth responses."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    full_prompt = f"""You are an expert AI assistant. Use the following context to answer the user's question in an elaborate and well-explained manner:
    
    Context:
    {context}
    
    User's Question:
    {prompt}

    Provide a detailed explanation with examples where necessary."""
    
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": full_prompt}]
    }
    
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json()}"

def generate_quiz(content, num_questions=3):
    """Generates quiz questions from extracted text using OpenRouter."""
    prompt = f"Generate {num_questions} multiple-choice questions based on the following content:\n{content}"
    return call_openrouter(prompt)

def chat_with_ai(user_input):
    """Interacts with AI chatbot using OpenRouter."""
    return call_openrouter(user_input)

def web_search(query):
    """Fetches top search results using DuckDuckGo."""
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    
    results = [entry["Text"] for entry in data.get("RelatedTopics", []) if "Text" in entry][:5]
    return "\n".join(results) if results else "No results found."

# --- UI Enhancements ---
st.set_page_config(page_title="AI Learning Assistant", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Upload PDF", "AI Assistant", "Generate Quiz"])

# --- PDF Upload Section ---
if page == "Upload PDF":
    st.title("Upload & Process PDF")

    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        st.info("Processing PDF...")
        text = extract_text_from_pdf(pdf_file)
        chunk_and_store_text(text)
        st.success("PDF successfully processed! You can now ask questions related to the document.")

    st.markdown("###Extracted Text (Preview)")
    if chunks:
        st.text_area("Extracted Text", "\n".join(chunks[:5]), height=200)

# --- AI Assistant Section ---
elif page == " AI Assistant":
    st.title("AI-Powered Research Assistant")

    query = st.text_input("Ask a Question:")
    if st.button("Get Answer"):
        with st.spinner("Fetching AI-powered response..."):
            relevant_text = retrieve_relevant_text(query)
            web_results = web_search(query)
            combined_context = f"Relevant text from PDF:\n{relevant_text}\n\nWeb search results:\n{web_results}"
            answer = call_openrouter(query, combined_context)
        
        st.success("Answer Generated!")
        with st.expander("AI Response"):
            st.write(answer)

    # --- Chatbot Section ---
    st.subheader("💬Chat with AI")
    chat_input = st.text_input("Enter your message:")
    if st.button("Send Message"):
        with st.spinner("Generating response..."):
            response = chat_with_ai(chat_input)
        st.success("Chat Response Received!")
        st.write("**Chatbot:**", response)

# --- Quiz Generation Section ---
elif page == "Generate Quiz":
    st.title("Generate AI-Powered Quiz")

    num_questions = st.number_input("How many questions?", min_value=1, max_value=20, value=5)
    if st.button("Generate Quiz"):
        if chunks:
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz("\n".join(chunks[:3]), num_questions)
            st.success("Quiz Generated!")
            with st.expander("Quiz Questions"):
                st.write(quiz)
        else:
            st.warning("Please upload a PDF first.")


