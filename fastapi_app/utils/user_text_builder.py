def _base_profile_lines(user) -> list[str]:
    lines = []
    city_or_region = getattr(user, "city", None) or getattr(user, "region", None)
    if city_or_region:
        lines.append(f"도시: {city_or_region}")
    if user.companion:
        lines.append("동행: " + ", ".join(user.companion))
    if user.activity_level:
        lines.append(f"활동강도: {user.activity_level}")
    if user.budget:
        lines.append(f"예산: {user.budget}")
    if user.avoid:
        lines.append("피하고 싶은 것: " + ", ".join(user.avoid))
    return lines


def build_tourspot_text(user) -> str:
    parts = _base_profile_lines(user)
    if user.preferred_themes:
        parts.append("선호 테마: " + ", ".join(user.preferred_themes))
    if user.preferred_moods:
        parts.append("선호 분위기: " + ", ".join(user.preferred_moods))
    history = getattr(user, "historyPlaces", None) or getattr(user, "visit_tourspot", None)
    if history:
        ts = [p for p in history if getattr(p, "category", None) == "tourspot"]
        if ts:
            parts.append(f"방문한 관광지 수: {len(ts)}")
    selected = getattr(user, "selectedPlaces", None) or getattr(user, "last_selected_pois", None)
    if selected:
        last_ts = [p for p in selected if getattr(p, "category", None) == "tourspot"]
        if last_ts:
            ids = [str(p.place_id) for p in last_ts if getattr(p, "place_id", None) is not None]
            parts.append("마지막 선택 관광지: " + ", ".join(ids))
    return ". ".join(parts)


def build_cafe_text(user) -> str:
    parts = _base_profile_lines(user)
    if user.preferred_cafe_types:
        parts.append("선호 카페 타입: " + ", ".join(user.preferred_cafe_types))
    if user.preferred_moods:
        parts.append("카페 분위기: " + ", ".join(user.preferred_moods))
    history = getattr(user, "historyPlaces", None) or getattr(user, "visit_cafe", None)
    if history:
        cafes = [p for p in history if getattr(p, "category", None) == "cafe"]
        if cafes:
            parts.append(f"방문한 카페 수: {len(cafes)}")
    selected = getattr(user, "selectedPlaces", None) or getattr(user, "last_selected_pois", None)
    if selected:
        last_cafe = [p for p in selected if getattr(p, "category", None) == "cafe"]
        if last_cafe:
            ids = [str(p.place_id) for p in last_cafe if getattr(p, "place_id", None) is not None]
            parts.append("마지막 선택 카페: " + ", ".join(ids))
    return ". ".join(parts)


def build_restaurant_text(user) -> str:
    parts = _base_profile_lines(user)
    if user.preferred_restaurant_types:
        parts.append("선호 음식점 타입: " + ", ".join(user.preferred_restaurant_types))
    history = getattr(user, "historyPlaces", None) or getattr(user, "visit_restaurant", None)
    if history:
        rests = [p for p in history if getattr(p, "category", None) == "restaurant"]
        if rests:
            parts.append(f"방문한 음식점 수: {len(rests)}")
    selected = getattr(user, "selectedPlaces", None) or getattr(user, "last_selected_pois", None)
    if selected:
        last_rest = [p for p in selected if getattr(p, "category", None) == "restaurant"]
        if last_rest:
            ids = [str(p.place_id) for p in last_rest if getattr(p, "place_id", None) is not None]
            parts.append("마지막 선택 음식점: " + ", ".join(ids))
    return ". ".join(parts)
