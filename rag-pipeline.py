from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# ✨ UPDATED: Import from the correct langchain-ollama package
from langchain_ollama import OllamaEmbeddings, ChatOllama

# 1. Load your data (Ensure "Quantum Computing.pdf" is in the same folder!)
import pypdf

pdf_path = "Quantum Computing.pdf"
reader = pypdf.PdfReader(pdf_path)

# Extract raw text page-by-page, creating clean LangChain document objects
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip(): # Ignore blank layout pages
        documents.append(Document(page_content=text, metadata={"page": i + 1}))

print(f"📚 Successfully extracted {len(documents)} pages from the PDF.")

# 2. Chunk the document into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"✂️ Split document into {len(chunks)} smaller text chunks.")

#chunks = [
#    Document(page_content="Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers."),
#    Document(page_content="Subatomic particles can exist in more than one state at the same time, which is called superposition.")
#]

# 3. ✨ UPDATED: Embed the chunks using the modern OllamaEmbeddings class
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)
#vector_store = Chroma.from_documents(chunks, embeddings)

# 4. Turn the vector store into a Retriever
#retriever = vector_store.as_retriever(search_kwargs={"k": 2}) 
print("⚙️ Initializing Vector Database...")
persist_directory = "./chroma_db"

# Create the initial database object using just the FIRST chunk to set it up
vector_store = Chroma.from_documents(
    documents=[chunks[0]], 
    embedding=embeddings, 
    persist_directory=persist_directory
)

# Load the remaining chunks in small batches of 100
batch_size = 100
print(f"📦 Commencing batch upload to protect Ollama memory thread...")

for i in range(1, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_store.add_documents(batch)
    print(f"✅ Processed chunks {i} to {min(i + batch_size, len(chunks))}...")

print("🎉 Vector database successfully constructed!")

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 5. ✨ UPDATED: Use ChatOllama for faster execution on Mac M4
llm = ChatOllama(
    model="llama3.2", 
    temperature=0,
    base_url="http://127.0.0.1:11434"
)

template = """You are a helpful assistant. Answer the question based ONLY on the following context:
{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Construct the RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Query the pipeline
query = "What is the main topic of the document?"
response = rag_chain.invoke(query)

print("\n--- AI Response ---")
print(response)