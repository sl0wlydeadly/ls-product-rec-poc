from typing import List, Dict
from pydantic import BaseModel

class PrefsRecommendRequest(BaseModel):
    customer_id: str
    preferences: Dict[str, List[str]]
    candidate_limit: int = 20
    top_k: int = 10
    exclude_bought: bool = True

class Product(BaseModel):
    id: str   # SKU
    title: str
    description: str
    tags: List[str]

class IndexRequest(BaseModel):
    items: List[Product]
