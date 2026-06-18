import streamlit as st
import os
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

# --- PAGE SETUP ---
st.set_page_config(page_title="Socratic Quantum Tutor", page_icon="🎓", layout="centered")
st.title("🎓 Socratic Quantum Tutor")
st.subheader("Master Quantum Computing through active reasoning.")

# --- CACHED RAG INITIALIZATION ---
# This decorator ensures the massive 1,348 chunks are only parsed once!
@st.cache_resource
def initialize_rag_pipeline():
    pdf_path = "Quantum Computing.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"❌ '{pdf_path}' not found in the directory! Please place it in the project folder.")
        st.stop()
        
    # 1. Parse PDF
    reader = pypdf.PdfReader(pdf_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip(): 
            documents.append(Document(page_content=text, metadata={"page": i + 1}))
            
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Vector Database Loading
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    persist_directory = "./chroma_db"
    
    vector_store = Chroma.from_documents(
        documents=[chunks[0]], 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    # Batch add chunks safely
    batch_size = 100
    for i in range(1, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        
    return vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize the retriever via the cache function
with st.status("🧠 Ingesting Quantum Textbook and structuring brain... please wait.", expanded=False) as status:
    retriever = initialize_rag_pipeline()
    status.update(label="🎉 Textbook fully processed! Ready to study.", state="complete")

# --- CONSTRUCT THE Socratic RAG CHAIN ---
llm = ChatOllama(model="llama3.2", temperature=0.7)

template = """You are a patient, encouraging, and brilliant Socratic tutor helping a student learn from their textbook. 

Your fundamental rule: NEVER give the answer away directly or simply paste definitions. 

Instead, follow this style:
1. Validate the student's curiosity or line of thinking.
2. Use the provided textbook context to explain the concept using a simple, real-world analogy or conceptual breakdown.
3. End your response by asking the student ONE targeted, thought-provoking question that prompts them to take the next logical step or apply what they just read.

Textbook Context:
{context}

Question: {question}
Socratic Tutor Response:"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- CHAT HISTORY TRACKING ---
# Streamlit clears its memory on every click/input. Session state preserves history.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": 'Hi! I am here to guide you with "Quantum Computing". Ask me anything, and let\'s break down the concepts together!'}
    ]

# Display historical messages in the web container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER CHAT INTERACTION ---
if user_query := st.chat_input("Ask a question (e.g., What is a qubit?)"):
    
    # 1. Show user's input instantly in the window
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Query the local RAG pipeline and show the Socratic tutor's answer
    with st.chat_message("assistant"):
        with st.spinner("Tutor is reflecting on the textbook context..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)
            
    # 3. Save response to session history
    st.session_state.messages.append({"role": "assistant", "content": response})