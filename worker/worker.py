import os
import time
import redis
import ollama
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 5))
LOG_FILE = os.getenv("LOG_FILE", "worker.log")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def connect_redis():
    for _ in range(5):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            logging.info("Connected to Redis.")
            return r
        except redis.ConnectionError:
            logging.warning("Redis not available, retrying...")
            time.sleep(2)
    raise ConnectionError("Could not connect to Redis.")

def process_question(question):
    """Send question to Ollama and return the response text."""
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": question}])
        return response['message']['content']
    except Exception as e:
        logging.error(f"Ollama error: {e}")
        return "Error generating answer."

def main():
    r = connect_redis()
    logging.info(f"Worker started using model {OLLAMA_MODEL}")

    while True:
        # Try to pop a question
        item = r.lpop("questions")
        if item:
            qid, question = item.split("|", 1)
            logging.info(f"Processing question {qid}: {question}")
            answer = process_question(question)

            # Store answer
            r.set(f"answer:{qid}", answer)
            r.lpush("history", f"{qid}|{question}|{answer}")
            logging.info(f"Answer stored for {qid}")
        else:
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
