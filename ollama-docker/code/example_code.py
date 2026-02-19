import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# ==============================
# PHASE 1: CONFIGURATION
# ==============================
# Ollama server URL (must be running locally).
OLLAMA_HOST = "http://localhost:11434"
# Embedding model used for vector search.
EMBED_MODEL = "nomic-embed-text"
# Chat model used to generate the final answer.
CHAT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
# Chroma collection name.
COLLECTION_NAME = "simple_knowledge"
# Number of chunks to retrieve for each question.
TOP_K = 2

# Keep all file paths relative to this script file.
BASE_DIR = Path(__file__).resolve().parent
ARTICLES_FILE = BASE_DIR / "articles.jsonl"
COUNTER_FILE = BASE_DIR / "counter.txt"
CHROMA_DIR = BASE_DIR / "chroma"

# ==============================
# PHASE 2: CLIENTS + COLLECTION
# ==============================
# Single Ollama client for both embedding and answer generation.
ollama_client = Client(host=OLLAMA_HOST)
# Persistent Chroma storage directory.
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# Create/load vector collection.
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ==============================
# PHASE 3: TEXT SPLITTING SETUP
# ==============================
# Split long article content into smaller chunks for embedding.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separators=[".", "\n"],
)


# ==============================
# PHASE 4: HELPER FUNCTIONS
# ==============================
def load_counter() -> int:
    # Read the last processed line index from counter file (resume support).
    if not COUNTER_FILE.exists():
        return 0
    try:
        raw = COUNTER_FILE.read_text(encoding="utf-8").strip()
        return int(raw) if raw else 0
    except ValueError:
        print("counter.txt has invalid value. Starting from line 0.")
        return 0


def save_counter(next_line_index: int) -> None:
    # Save the next line index to process.
    COUNTER_FILE.write_text(str(next_line_index), encoding="utf-8")


def build_or_resume_vector_db() -> None:
    # Build embeddings from articles.jsonl, resuming from counter.txt.
    if not ARTICLES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ARTICLES_FILE}")

    start_line = load_counter()
    print(f"Building database from line {start_line}...")

    with ARTICLES_FILE.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            # Skip lines already processed in previous runs.
            if i < start_line:
                continue

            # Parse one JSON line (one article object).
            article = json.loads(line)
            content = article.get("content", "").strip()
            title = article.get("title", "Untitled")

            # If there is no content, mark line as done and continue.
            if not content:
                save_counter(i + 1)
                continue

            # Split article text into clean chunks.
            chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]

            # Embed each chunk and upsert into Chroma.
            for j, chunk in enumerate(chunks):
                embed_response = ollama_client.embed(
                    model=EMBED_MODEL,
                    input=f"search_document: {chunk}",
                )
                embedding = embed_response["embeddings"][0]

                collection.upsert(
                    ids=[f"id_{i}_{j}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"line": i, "chunk": j, "title": title}],
                )

            # Persist progress after each processed line.
            save_counter(i + 1)
            print(f"Processed line {i}")

    print("Database build/resume completed.")


def retrieve_context(question: str, n_results: int = TOP_K) -> str:
    # Convert user question to embedding vector.
    query_embedding = ollama_client.embed(
        model=EMBED_MODEL,
        input=f"query: {question}",
    )["embeddings"][0]

    # Retrieve top matching chunks from vector DB.
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    retrieved_docs = results.get("documents", [[]])[0]

    # Merge docs into one context block for prompt.
    context = "\n\n".join(doc for doc in retrieved_docs if doc)
    return context


def generate_answer(question: str, context: str) -> str:
    # Build prompt: model should answer only from retrieved context.
    prompt = f"""You are a helpful assistant.
Answer only from the provided context.
If context does not contain enough information, reply exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    # Ask chat model to generate final answer.
    response = ollama_client.generate(
        model=CHAT_MODEL,
        prompt=prompt,
        options={"temperature": 0.1},
    )

    # Return model text safely.
    answer = response.get("response", "").strip()
    return answer if answer else "I don't know."


def run_chat_loop() -> None:
    # Interactive chatbot loop.
    print("Chatbot is ready. Type 'exit' to stop.")

    while True:
        user_input = input("How may I assist you? ").strip()

        # Exit commands for clean stop.
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        # Ignore empty messages.
        if not user_input:
            print("Please type a question.")
            continue

        # Retrieve context for current user question.
        context = retrieve_context(user_input, n_results=TOP_K)
        # Generate answer from retrieved context + question.
        answer = generate_answer(user_input, context)

        print("\nAnswer:")
        print(answer)
        print("-" * 60)


# ==============================
# PHASE 5: PROGRAM ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Step 1: Build or resume the vector database.
    build_or_resume_vector_db()
    # Step 2: Start chatbot loop.
    run_chat_loop()
