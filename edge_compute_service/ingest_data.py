import os
import weaviate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
DATA_DIR = "/app/knowledge_base"

client = weaviate.connect_to_custom(
    http_host="weaviate",      # The service name in docker-compose
    http_port=8080,
    http_secure=False,         # No SSL inside local network
    grpc_host="weaviate",      # The gRPC service name
    grpc_port=50051,           # The gRPC port (default)
    grpc_secure=False,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest", 
    base_url=OLLAMA_BASE_URL
)

def get_security_metadata(file_path):

    folder_name = os.path.basename(os.path.dirname(file_path))
    
    metadata = {}
    
    if folder_name == "admin":
        metadata = {"access_level": "admin"}
    elif folder_name == "public_oakhillpines":
        metadata = {"access_level": "public_oakhillpines"}
    elif folder_name == "family":
        metadata = {"access_level": "family"}
    elif folder_name == "private_oakhillpines":
        metadata = {"access_level": "private_oakhillpines"}
    else:
        metadata = {"access_level": "general"}
        
    print(f"Processing: {os.path.basename(file_path)} -> Tags: {metadata}")
    return metadata

def main():
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="PermanentKnowledge")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found inside container.")
        return

    print("Parsing PDFs from disk...")
    
    # handles .pdf, .txt, .docx
    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        recursive=True,
        file_metadata=get_security_metadata
    ).load_data()

    print(f"Loaded {len(documents)} pages/chunks. Indexing now (this uses GPU)...")

    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    
    print("Success! All PDFs are now indexed with security metadata.")

if __name__ == "__main__":
    main()