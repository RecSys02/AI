def _base_profile_lines(user) -> list[str]:
    lines = []
    if user.city:
        lines.append(f"도시: {user.city}")
    if user.companion_type:
        lines.append("동행: " + ", ".join(user.companion_type))
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
    if user.visit_tourspot:
        parts.append(f"방문한 관광지 수: {len(user.visit_tourspot)}")
    if user.last_selected_poi and getattr(user.last_selected_poi, "category", None) == "tourspot":
        parts.append(f"마지막 선택 관광지: {user.last_selected_poi.region}-{user.last_selected_poi.id}")
    return ". ".join(parts)


def build_cafe_text(user) -> str:
    parts = _base_profile_lines(user)
    if user.preferred_cafe_types:
        parts.append("선호 카페 타입: " + ", ".join(user.preferred_cafe_types))
    if user.preferred_moods:
        parts.append("카페 분위기: " + ", ".join(user.preferred_moods))
    if user.visit_cafe:
        parts.append(f"방문한 카페 수: {len(user.visit_cafe)}")
    if user.last_selected_poi and getattr(user.last_selected_poi, "category", None) == "cafe":
        parts.append(f"마지막 선택 카페: {user.last_selected_poi.region}-{user.last_selected_poi.id}")
    return ". ".join(parts)


def build_restaurant_text(user) -> str:
    parts = _base_profile_lines(user)
    if user.preferred_restaurant_types:
        parts.append("선호 음식점 타입: " + ", ".join(user.preferred_restaurant_types))
    if user.preferred_moods:
        parts.append("식사 분위기: " + ", ".join(user.preferred_moods))
    if user.visit_restaurant:
        parts.append(f"방문한 음식점 수: {len(user.visit_restaurant)}")
    if user.last_selected_poi and getattr(user.last_selected_poi, "category", None) == "restaurant":
        parts.append(f"마지막 선택 음식점: {user.last_selected_poi.region}-{user.last_selected_poi.id}")
    return ". ".join(parts)
