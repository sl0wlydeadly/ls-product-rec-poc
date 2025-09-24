# POC Recommender (FastAPI + Qdrant + Embeddings + LLM)

A minimal, clean recommender PoC:
- **Embeddings** (via **Ollama**) to index and retrieve similar products.
- **Qdrant** as the vector DB.
- **LLM** (OpenAI-compatible **Llama Stack**) to craft short, human-friendly suggestion texts.
- Two endpoints:
  - `/index` to index products into Qdrant.
  - `/recommend/prefs_points` deterministic scoring.
  - `/recommend/prefs_llm` suggestion texts using LLM + Qdrant candidates.

---

## 1) Structure (all files in the root folder)

```
main.py
config.py
schemas.py
embeddings.py
qdrant.py
scoring.py
llm_client.py
recommender_points.py
recommender_llm.py
Dockerfile
docker-compose.yml
README.md
```

---

## 2) Environment

These env vars are read by the app (defaults shown):

```
LOG_LEVEL=INFO
DEBUG=0

OLLAMA_URL=http://ollama:11434
EMBED_MODEL=nomic-embed-text

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=products_poc

LLAMA_STACK_URL=http://llama-stack:8080
MODEL_ID=llama3.2:3b      # Your LLM id (OpenAI-compatible server)

RECO_MAX_RESULTS=10
RECO_SCORE_THRESHOLD=0.01
```

Create a `.env` file (optional) or set envs in `docker-compose.yml`.

---

## 3) Quick start with Docker Compose

**Example `docker-compose.yml`**:

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=24h

  llama-stack:
    image: vllm/vllm-openai:latest
    container_name: llama-stack
    ports:
      - "8080:8000"
    command:
      - --model
      - meta-llama/Llama-3.2-3B-Instruct
      - --dtype
      - float16
      - --max-model-len
      - "4096"

  recommender:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - GIT_SHA=${GIT_SHA:-dev}
    container_name: recommender
    depends_on:
      - qdrant
      - ollama
      - llama-stack
    environment:
      - LOG_LEVEL=INFO
      - DEBUG=0
      - OLLAMA_URL=http://ollama:11434
      - EMBED_MODEL=nomic-embed-text
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=products_poc
      - LLAMA_STACK_URL=http://llama-stack:8000
      - MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
      - RECO_MAX_RESULTS=10
      - RECO_SCORE_THRESHOLD=0.01
    ports:
      - "9000:9000"

volumes:
  qdrant_data:
  ollama_data:
```

### Build & run

```bash
docker compose up -d --build
```

Check containers:

```bash
docker compose ps
```

---

## 4) Models: pulling / preparing

### 4.1 Embedding model (Ollama)

Pull the embedding model (once):

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

Test it:

```bash
curl -s http://localhost:11434/api/embeddings   -H "Content-Type: application/json"   -d '{"model":"nomic-embed-text","prompt":"hello world"}' | jq
```

### 4.2 LLM model (“Llama Stack” / OpenAI-compatible)

If you use the **vLLM** service in the compose file above, it will **download** `meta-llama/Llama-3.2-3B-Instruct` on first start.

List models:

```bash
curl -s http://localhost:8080/v1/models | jq
```

Test chat:

```bash
curl -s http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model":"meta-llama/Llama-3.2-3B-Instruct",
        "messages":[{"role":"user","content":"Say hi in 5 words."}],
        "temperature":0.0,
        "max_tokens":50
      }' | jq
```

---

## 5) Running the API

Once services are up:

```bash
curl http://localhost:9000/openapi.json | jq .info
```

---

## 6) Endpoints

### 6.1 `POST /index`

Index products into Qdrant.

**Body**
```json
{
  "items": [
    {
      "id": "sku-033",
      "title": "Mountain Bike",
      "description": "A durable MTB with 27.5\" wheels.",
      "tags": ["cycling", "bike", "outdoors"]
    }
  ]
}
```

**Example**
```bash
curl -s http://localhost:9000/index   -H "Content-Type: application/json"   -d @products.json | jq
```

---

### 6.2 `POST /recommend/prefs_points`

Deterministic, rule-based ranking.

**Body**
```json
{
  "customer_id": "user-123",
  "preferences": {
    "clicked": ["sku-033"],
    "added_to_cart": ["sku-046"],
    "bought": []
  },
  "candidate_limit": 20,
  "top_k": 10,
  "exclude_bought": true
}
```

---

### 6.3 `POST /recommend/prefs_llm`

LLM suggestions (human-friendly CTA).

**Body**
```json
{
  "customer_id": "user-456",
  "preferences": {
    "clicked": ["sku-033"],
    "added_to_cart": ["sku-046"],
    "bought": ["sku-015"]
  },
  "candidate_limit": 20,
  "top_k": 5,
  "exclude_bought": true
}
```

**Response (shape)**
```json
{
  "customer_id": "user-456",
  "suggestions": [
    {
      "text": "You viewed “Mountain Bike” — check out “Cycling Helmet”.",
      "source_sku": "sku-033",
      "target_sku": "sku-031"
    }
  ]
}
```

---

## 7) Rebuild / update

Rebuild if code changes:

```bash
docker compose up -d --build recommender
```

Restart only the API:

```bash
docker compose restart recommender
```

---

## 8) Troubleshooting

- **Model not found**: pull `nomic-embed-text` into Ollama, verify `MODEL_ID` exists in `/v1/models`.
- **Empty recommendations**: index products first (`/index`).
- **Connectivity**: test from inside container:
  ```bash
  docker exec -it recommender sh
  wget -qO- http://qdrant:6333/collections
  wget -qO- http://ollama:11434/api/tags
  wget -qO- http://llama-stack:8000/v1/models
  ```

---

## 9) Version & health

Check FastAPI version in OpenAPI:

```bash
curl -s http://localhost:9000/openapi.json | jq .info.version
```

