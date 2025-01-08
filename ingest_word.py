from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'word/'
DB_FAISS_PATH = 'vectorstore/db_faiss'  # Path to the single Word file

def create_vector_db():
    print("Loading document...")
    loader = DirectoryLoader(DATA_PATH, glob=['*.docx','*.doc'], loader_cls=Docx2txtLoader)
    documents = loader.load()

    print("Loaded documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)  # Wrapping document in a list for splitting

    print(f"Split document into {len(texts)} chunks.")
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    
    print(f"Saving vector store to {DB_FAISS_PATH}...")
    db.save_local(DB_FAISS_PATH)
    print("Vector store saved successfully.")

if __name__ == "__main__":
    create_vector_db()
