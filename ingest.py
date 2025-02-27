# import os 
# import time
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from models import Models
# from uuid import uuid4

# load_dotenv()

# #Initialize the model
# models = Models()
# embeddings = models.embeddings_ollama
# llm = models.model_ollama

# #define constants
# data_folder = "./data"
# chunk_size = 1000
# chunk_overlap = 50
# chunk_interval = 10

# #chroma vectorstore
# vector_store = Chroma(
#     collection_name="documents",
#     embedding_function=embeddings,
#     persist_directory="./db/chroma_langchain_db", ##locally data will be saved here
# )

# #ingest a file
# def ingest_file(file_path):
#     #skip the non-pdf files
#     if not file_path.endswith(".pdf"):
#         print(f"Skipping {file_path} as it is not a PDF file")
#         return
#     print(f"Ingesting {file_path}")    
#     #load the document
#     loader = PyPDFLoader(file_path)
#     loaded_documents = loader.load()
#     #split the document
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=[".", "!", "?", "\n"],
#     )
    
#     documents = text_splitter.split_documents(loaded_documents)
#     uuids = [str(uuid4()) for _ in range(len(documents))]
#     print(f"Adding {len(documents)} chunks to the vector store")
#     vector_store.add_documents(documents=documents, id=uuids)
#     print(f"finished ingesting {file_path}")

# #Main loop
# def main_loop():
#     while True:
#         for filename in os.listdir(data_folder):
#             if not filename.startswith("_"):
#                 file_path = os.path.join(data_folder, filename)
#                 ingest_file(file_path)
#                 new_filename = "_" + filename
#                 new_file_path = os.path.join(data_folder, new_filename)
#                 os.rename(file_path, new_file_path)
#         time.sleep(chunk_interval)        #check the folder every 10 seconds

# #run the main loop
# if __name__ == "__main__":
#     main_loop()       

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import Models
from uuid import uuid4

# Load environment variables
load_dotenv()

# Initialize the model
models = Models()
embeddings = models.embeddings_ollama  # Ensure Ollama embeddings are working
llm = models.model_ollama  # Ensure the model is properly initialized

# Define constants
CHUNK_SIZE = 5000  # Increased chunk size for better retrieval
CHUNK_OVERLAP = 100
DATA_FOLDER = "./data"  # Folder where PDFs are stored
DB_DIRECTORY = "./db/chroma_langchain_db"  # ChromaDB storage path

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=DB_DIRECTORY,
)

# Function to ingest a single file
def ingest_file(file_path):
    if not file_path.endswith(".pdf"):
        print(f"Skipping {file_path} (not a PDF)")
        return

    print(f"\n📄 Ingesting {file_path}...")

    # Load the document using PDFPlumber (better text extraction)
    loader = PDFPlumberLoader(file_path)
    loaded_documents = loader.load()

    if not loaded_documents:
        print(f"⚠️ No text extracted from {file_path}. Check the PDF format.")
        return

    print(f"✅ Loaded {len(loaded_documents)} pages from {file_path}")

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[".", "!", "?", "\n"],
    )
    
    documents = text_splitter.split_documents(loaded_documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]

    print(f"🔹 Splitting completed: {len(documents)} chunks generated.")

    # Add documents to ChromaDB
    vector_store.add_documents(documents=documents, ids=uuids)

    print(f"✅ Finished ingesting {file_path}")

# Function to check stored chunks
def check_chroma_db():
    print("\n🔍 Verifying ChromaDB content...")
    count = vector_store._collection.count()
    print(f"📊 Total stored document chunks in ChromaDB: {count}")

# Main function to ingest all PDFs in the folder
def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"⚠️ Data folder '{DATA_FOLDER}' does not exist. Create it and add PDFs.")
        return

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDF files found in '{DATA_FOLDER}'. Please add PDFs.")
        return

    for filename in pdf_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        ingest_file(file_path)

    # Check ChromaDB storage after ingestion
    check_chroma_db()

if __name__ == "__main__":
    main()

