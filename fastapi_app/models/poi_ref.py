# models/poi_ref.py
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])

class PoiRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    place_id: int
    category: str      # "tourspot" | "restaurant" | "cafe"
    province: str        # "seoul", "busan", ...
    visited_at: datetime | None = Field(default=None, alias="visitedAt")
