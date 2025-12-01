import os
import time
import json
import redis
import weaviate

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

global_index = None

def init_global_index():
    global global_index
    
    client = weaviate.connect_to_custom(
        http_host="weaviate",      
        http_port=8080,
        http_secure=False,         
        grpc_host="weaviate",      
        grpc_port=50051,           
        grpc_secure=False,
    )
    
    Settings.llm = Ollama(
        model=OLLAMA_MODEL, 
        base_url=OLLAMA_BASE_URL, 
        request_timeout=100.0
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest", 
        base_url=OLLAMA_BASE_URL
    )

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="PermanentKnowledge")
    
    global_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

def process_rag_request(question, auth_params):
    if isinstance(auth_params, str):
        auth_params = [auth_params]
    
    if not auth_params:
        return "Error: Access Denied."

    acl_filters = []
    for param in auth_params:
        acl_filters.append(MetadataFilter(
            key="access_level",
            value=param,
            operator=FilterOperator.EQ
        ))

    if len(acl_filters) > 1:
        secure_filters = MetadataFilters(
            filters=acl_filters, 
            condition="or"
        )
    else:
        secure_filters = MetadataFilters(filters=acl_filters)

    jit_engine = global_index.as_query_engine(
        filters=secure_filters,
        similarity_top_k=3
    )

    response = jit_engine.query(question)
    return str(response)

def process_direct_request(question):
    response = Settings.llm.complete(question)
    return str(response)

def main():
    init_global_index()
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    while True:
        item = r.lpop("questions")
        if item:
            try:
                parts = item.split("|", 2)
                
                if len(parts) == 3:
                    qid, question, params = parts
                    try:
                        domain_params = json.loads(params)
                    except json.JSONDecodeError:
                        domain_params = [params]
                else:
                    qid, question = parts[0], parts[1]
                    domain_params = [] 
                
                # answer = process_rag_request(question, domain_params)
                answer = process_direct_request(question)

                r.set(f"answer:{qid}", answer)
                
            except Exception as e:
                print(f"Redis Error: {e}", flush=True)
        else:
            time.sleep(3)

if __name__ == "__main__":
    main()