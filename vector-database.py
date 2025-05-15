import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Directories
pdfs_directory = 'pdfs/'
vectorstore_directory = 'vectorstore/'

# Create directories if not exist
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs(vectorstore_directory, exist_ok=True)

# Upload & Load PDF
def upload_pdf(file):
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = 'pdfs/universal_declaration_of_human_rights.pdf'
# file_path = 'pdfs/'

documents = load_pdf(file_path)

# Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,   # Correct spelling now
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)


# setup embeddings model by Deepseek R1 with Ollama

ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings

embeddings = get_embedding_model(ollama_model_name)

# create embeding and store in FAISS Vector Store

FAISS_DB_PATH = os.path.join(vectorstore_directory, "db_faiss")
faiss_db = FAISS.from_documents(text_chunks, embeddings)
faiss_db.save_local(FAISS_DB_PATH)

