# models/poi_ref.py
from pydantic import BaseModel

class PoiRef(BaseModel):
    place_id: int
    category: str      # "tourspot" | "restaurant" | "cafe"
    province: str        # "seoul", "busan", ...
