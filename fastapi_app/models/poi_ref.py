# models/poi_ref.py
from pydantic import BaseModel

class PoiRef(BaseModel):
    id: int
    category: str      # "tourspot" | "restaurant" | "cafe"
    region: str        # "seoul", "busan", ...
