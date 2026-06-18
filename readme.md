# 🎓 Socratic Study Buddy: Quantum Computing Edition

An interactive, web-based AI learning tutor powered by a local **RAG (Retrieval-Augmented Generation)** pipeline. Instead of just giving away definitions, this application uses a textbook document to answer student questions using **Socratic methodology**—guiding students through complex concepts via custom analogies, structured insights, and thought-provoking follow-up questions.

The entire application runs **locally and 100% free** on your computer, with no API keys or cloud dependencies needed.

---

## 🚀 Features
- **Socratic Persona:** Shuts down dry copy-pasted definitions. Instructs the AI model to explain terms dynamically and challenge the user back.
- **Local Ingestion & Batching:** Pages are safely stripped from target PDFs and trickled into the database to stay completely lightweight.
- **Instant Browser Chat UI:** A responsive, ChatGPT-style layout designed natively with Streamlit.
- **Smart Memory Caching:** The 1,300+ document chunks are processed and vectorized only on the initial spin-up, preventing repetitive performance lag.

---

## 🛠️ System Prerequisites

### 1. Python 3.x
Ensure you have a modern version of Python running. On macOS, check this via terminal:
```bash
python3 --version
```
### 2. Download local AI models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
### 3. Install Project Dependencies
```bash
pip3 install streamlit langchain langchain-ollama langchain-chroma pypdf
```
## To run the application use:
```bash
streamlit run app.py
```
