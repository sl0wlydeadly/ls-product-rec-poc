# POC Recommender (FastAPI + Qdrant + Embeddings + LLM)

A minimal, clean recommender PoC:
- **Embeddings** (via **Ollama**) to index and retrieve similar products.
- **Qdrant** as the vector DB.
- **LLM** (OpenAI-compatible **Llama Stack**) to craft short, human-friendly suggestion texts.
- Three endpoints:
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

LLAMA_STACK_URL=http://llama-stack:8080/v1/openai
MODEL_ID=llama3.2:3b

RECO_MAX_RESULTS=10
RECO_SCORE_THRESHOLD=0.01
```

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
docker exec -it ollama ollama pull llama3.2:3b

# Download the nomic embed text model
docker exec -it ollama ollama pull nomic-embed-text:latest

Test it:

```bash
curl -s http://localhost:11434/api/embeddings   -H "Content-Type: application/json"   -d '{"model":"nomic-embed-text","prompt":"hello world"}' | jq
```

### 4.2 LLM model (“Llama Stack” / OpenAI-compatible)
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
### Full example at the end of this README
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

## Indexing the full list of products for this PoC

```
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
  ```
  curl -s http://localhost:9000/index \
  -H "Content-Type: application/json" \
  --data-raw '{
    "items":[
      {"id":"sku-011","title":"Laptop Ultra","description":"14-inch ultrabook 16GB RAM","tags":["electronics","laptop","portable","work","office"]},
      {"id":"sku-012","title":"Gaming Mouse X","description":"Ergonomic RGB gaming mouse","tags":["electronics","gaming","mouse","peripheral","rgb"]},
      {"id":"sku-013","title":"Mechanical Keyboard","description":"RGB mechanical keyboard","tags":["electronics","keyboard","mechanical","gaming","rgb"]},
      {"id":"sku-014","title":"External SSD 1TB","description":"USB-C rugged external SSD","tags":["electronics","storage","ssd","portable","usb-c"]},
      {"id":"sku-015","title":"Smartphone Pro Max","description":"6.7 inch AMOLED phone","tags":["electronics","smartphone","mobile","camera","5g"]},
      {"id":"sku-016","title":"Tablet Sketch","description":"10-inch tablet for drawing","tags":["electronics","tablet","drawing","stylus","portable"]},
      {"id":"sku-017","title":"Bluetooth Speaker","description":"Portable waterproof speaker","tags":["electronics","audio","bluetooth","waterproof","portable"]},
      {"id":"sku-018","title":"4K Monitor","description":"27-inch UHD monitor HDR","tags":["electronics","monitor","display","4k","hdr"]},
      {"id":"sku-019","title":"VR Headset","description":"Immersive VR headset with controllers","tags":["electronics","vr","gaming","immersive","headset"]},
      {"id":"sku-020","title":"Drone Explorer","description":"4K drone with GPS and gimbal","tags":["electronics","drone","camera","flying","gps"]}
    ]
  }'
  ```
  ```
  curl -s http://localhost:9000/index \
  -H "Content-Type: application/json" \
  --data-raw '{
    "items":[
      {"id":"sku-021","title":"Cookware Set","description":"Non-stick 10-piece cookware","tags":["home","kitchen","cookware","non-stick","set"]},
      {"id":"sku-022","title":"Vacuum Cleaner","description":"Bagless cyclonic vacuum","tags":["home","cleaning","vacuum","bagless","cyclonic"]},
      {"id":"sku-023","title":"Air Purifier","description":"HEPA filter air purifier","tags":["home","air","purifier","hepa","filter"]},
      {"id":"sku-024","title":"Smart Light Bulbs","description":"WiFi RGB smart bulbs set","tags":["home","lighting","smart","wifi","rgb"]},
      {"id":"sku-025","title":"Coffee Maker","description":"Espresso machine with milk frother","tags":["home","coffee","espresso","machine","kitchen"]},
      {"id":"sku-026","title":"Standing Desk","description":"Adjustable height desk","tags":["office","desk","adjustable","standing","ergonomic"]},
      {"id":"sku-027","title":"Ergo Chair","description":"Ergonomic office chair","tags":["office","chair","ergonomic","comfort","work"]},
      {"id":"sku-028","title":"Bookshelf Tall","description":"5-shelf oak bookcase","tags":["furniture","bookshelf","storage","wood","tall"]},
      {"id":"sku-029","title":"Sofa Comfort","description":"3-seat modern sofa","tags":["furniture","sofa","livingroom","comfort","modern"]},
      {"id":"sku-030","title":"Bed Frame Queen","description":"Sturdy wooden bed frame","tags":["furniture","bed","frame","wood","queen"]}
    ]
  }'
  ```
  ```
  curl -s http://localhost:9000/index \
  -H "Content-Type: application/json" \
  --data-raw '{
    "items":[
      {"id":"sku-031","title":"Cycling Helmet","description":"Aerodynamic road bike helmet","tags":["cycling","helmet","safety","bike","gear"]},
      {"id":"sku-032","title":"Road Bike Shoes","description":"Clipless cycling shoes","tags":["cycling","shoes","bike","road","clipless"]},
      {"id":"sku-033","title":"Mountain Bike","description":"Full suspension MTB","tags":["cycling","mountain","bike","suspension","trail"]},
      {"id":"sku-034","title":"Bike Lights Set","description":"Front and rear USB lights","tags":["cycling","lights","bike","safety","usb"]},
      {"id":"sku-035","title":"Bike Repair Kit","description":"Compact multi-tool and patches","tags":["cycling","repair","tool","bike","kit"]},
      {"id":"sku-036","title":"Treadmill Pro","description":"Foldable treadmill with incline","tags":["fitness","treadmill","running","indoor","cardio"]},
      {"id":"sku-037","title":"Rowing Machine","description":"Magnetic resistance rower","tags":["fitness","rowing","machine","indoor","cardio"]},
      {"id":"sku-038","title":"Dumbbell Set","description":"Adjustable dumbbells pair","tags":["fitness","dumbbell","weights","adjustable","strength"]},
      {"id":"sku-039","title":"Pull-Up Bar","description":"Doorframe pull-up bar","tags":["fitness","pull-up","strength","indoor","exercise"]},
      {"id":"sku-040","title":"Resistance Bands","description":"Set of 5 resistance bands","tags":["fitness","bands","exercise","portable","workout"]}
    ]
  }'
  ```
  ```
  curl -s http://localhost:9000/index \
  -H "Content-Type: application/json" \
  --data-raw '{
    "items":[
      {"id":"sku-041","title":"Novel: Fantasy Quest","description":"Epic fantasy adventure novel","tags":["book","fantasy","adventure","novel","story"]},
      {"id":"sku-042","title":"Novel: Sci-Fi Galaxy","description":"Space opera science fiction novel","tags":["book","sci-fi","galaxy","space","novel"]},
      {"id":"sku-043","title":"Cookbook: Vegan Meals","description":"Plant-based recipe collection","tags":["book","cookbook","vegan","recipe","food"]},
      {"id":"sku-044","title":"Biography: Innovator","description":"Biography of tech innovator","tags":["book","biography","technology","innovation","life"]},
      {"id":"sku-045","title":"Self-Help: Mindset","description":"Guide to positive thinking","tags":["book","self-help","mindset","psychology","growth"]},
      {"id":"sku-046","title":"Jacket WinterPro","description":"Insulated winter jacket","tags":["fashion","jacket","winter","clothing","warm"]},
      {"id":"sku-047","title":"Sneakers Urban","description":"Streetwear sneakers","tags":["fashion","sneakers","urban","style","comfort"]},
      {"id":"sku-048","title":"Jeans SlimFit","description":"Slim fit blue jeans","tags":["fashion","jeans","denim","clothing","style"]},
      {"id":"sku-049","title":"T-Shirt Classic","description":"100% cotton classic t-shirt","tags":["fashion","tshirt","casual","clothing","basic"]},
      {"id":"sku-050","title":"Sunglasses UVShield","description":"Polarized sunglasses","tags":["fashion","sunglasses","uv","protection","style"]}
    ]
  }'
```