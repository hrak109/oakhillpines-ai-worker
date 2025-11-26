import os
import time
import json
import redis
import weaviate
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

global_index = None

def init_global_index():

    global global_index
    
    client = weaviate.connect_to_custom(
        http_host="weaviate",      # The service name in docker-compose
        http_port=8080,
        http_secure=False,         # No SSL inside local network
        grpc_host="weaviate",      # The gRPC service name
        grpc_port=50051,           # The gRPC port (default)
        grpc_secure=False,
    )
    
    Settings.llm = Ollama(model="llama3.2:3b", request_timeout=100.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="PermanentKnowledge")
    
    global_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

def process_secure_request(question, auth_params):

    try:
        filters_list = []
        
        if "access_level" in auth_params:
            filters_list.append(MetadataFilter(
                key="access_level", 
                value=auth_params["access_level"], 
                operator=FilterOperator.EQ
            ))
            
        # if "domain" in auth_params:
        #     filters_list.append(MetadataFilter(
        #         key="domain", 
        #         value=auth_params["domain"], 
        #         operator=FilterOperator.EQ
        #     ))

        secure_filters = MetadataFilters(filters=filters_list, condition="and")

        jit_engine = global_index.as_query_engine(
            filters=secure_filters,
            similarity_top_k=3
        )
        
        return str(jit_engine.query(question))
        
    except Exception as e:
        print(f"Secure Inference Error: {e}")
        return "Error: No authorized domain access."

def main():
    init_global_index()
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    while True:
        item = r.lpop("questions")
        if item:
            try:
                parts = item.split("|", 2)
                if len(parts) == 3:
                    qid, question, json_params = parts
                    try:
                        auth_params = json.loads(json_params)
                    except:
                        auth_params = {}
                else:
                    qid, question = parts[0], parts[1]
                    auth_params = {} 

                print(f"Processing {qid} with params: {auth_params}")
                
                answer = process_secure_request(question, auth_params)

                r.set(f"answer:{qid}", answer)
                
            except Exception as e:
                print(f"Redis Error: {e}")
        else:
            time.sleep(2)

if __name__ == "__main__":
    main()