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

Make sure you have the correct models available in your Ollama instance.  
Run the following commands to pull them:

```bash
# Download the LLaMA 3.2 3B model
ollama pull llama3.2:3b

# Download the nomic embed text model
ollama pull nomic-embed-text:latest

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
curl -s http://localhost:8080/v1/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [
      {"role":"system","content":"You are a tester."},
      {"role":"user","content":"Say OK"}
    ],
    "max_tokens": 8,
    "temperature": 0
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
curl -s http://localhost:9000/index \
  -H "Content-Type: application/json" \
  --data-raw '{
    "items":[
      {"id":"sku-001","title":"Trailblazer Backpack","description":"50L backpack with hydration system","tags":["outdoor","backpack","hiking","travel","lightweight"]},
      {"id":"sku-002","title":"Summit Tent","description":"2-person ultralight tent","tags":["outdoor","tent","camping","shelter","lightweight"]},
      {"id":"sku-003","title":"Mountain Sleeping Bag","description":"Down sleeping bag rated -10C","tags":["outdoor","sleeping","bag","insulation","winter"]},
      {"id":"sku-004","title":"Solar Charger Pro","description":"Portable 20W solar panel charger","tags":["electronics","solar","charger","outdoor","portable"]},
      {"id":"sku-005","title":"NoiseCancel Pro Headphones","description":"Over-ear wireless headphones","tags":["electronics","audio","wireless","noise-cancelling","music"]},
      {"id":"sku-006","title":"Smartwatch Fit","description":"Fitness smartwatch with GPS","tags":["electronics","wearable","fitness","gps","tracking"]},
      {"id":"sku-007","title":"Running Shoes Pro","description":"Lightweight cushioned running shoes","tags":["shoes","running","fitness","lightweight","comfort"]},
      {"id":"sku-008","title":"Climbing Rope","description":"Dynamic rope 60m UIAA certified","tags":["outdoor","climbing","rope","safety","gear"]},
      {"id":"sku-009","title":"Yoga Mat Deluxe","description":"Eco-friendly non-slip yoga mat","tags":["fitness","yoga","mat","eco","workout"]},
      {"id":"sku-010","title":"Hiking Poles","description":"Adjustable carbon trekking poles","tags":["outdoor","hiking","trekking","lightweight","support"]}
    ]
  }'
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

