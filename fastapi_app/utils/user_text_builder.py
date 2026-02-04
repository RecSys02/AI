def _base_profile_lines(user, include_activity: bool = True) -> list[str]:
    lines = []
    city_or_region = getattr(user, "city", None) or getattr(user, "region", None)
    if city_or_region:
        lines.append(f"도시: {city_or_region}")
    if user.companion:
        lines.append("동행: " + ", ".join(user.companion))
    if include_activity and user.activity_level:
        lines.append(f"활동강도: {user.activity_level}")
    if user.budget:
        lines.append(f"예산: {user.budget}")
    return lines


def build_tourspot_text(user) -> str:
    parts = _base_profile_lines(user, include_activity=True)
    if user.preferred_themes:
        parts.append("선호 테마: " + ", ".join(user.preferred_themes))
    if user.preferred_moods:
        parts.append("선호 분위기: " + ", ".join(user.preferred_moods))
    return ". ".join(parts)


def build_cafe_text(user) -> str:
    parts = _base_profile_lines(user, include_activity=False)
    if user.preferred_cafe_types:
        # 쉼표로 구분된 경우 분리 처리
        all_preferences = []
        for pref in user.preferred_cafe_types:
            pref_items = [p.strip() for p in pref.split(',')]
            all_preferences.extend(pref_items)

        # 선호 타입은 한 번만 명시해 과도한 반복을 방지
        preference_text = "선호 카페 타입: " + ", ".join(all_preferences)
        parts.append(preference_text)

    if user.preferred_moods:
        parts.append("카페 분위기: " + ", ".join(user.preferred_moods))
    return ". ".join(parts)


def build_restaurant_text(user) -> str:
    parts = _base_profile_lines(user, include_activity=False)
    if user.preferred_restaurant_types:
        # POI 임베딩 형식에 맞춰 "세계음식 > 양식" 형태로 추가
        food_type_mapping = {
            "한식": "한국음식",
            "한국음식": "한국음식",
            "중식": "중식",
            "중국음식": "중식",
            "일식": "일식",
            "일본음식": "일식",
            "양식": "양식",
            "서양음식": "양식",
        }

        # 쉼표로 구분된 경우 분리 처리
        all_preferences = []
        for pref in user.preferred_restaurant_types:
            pref_items = [p.strip() for p in pref.split(',')]
            all_preferences.extend(pref_items)

        # 선호 타입은 한 번만 명시해 과도한 반복을 방지
        preference_text = "선호 음식점 타입: " + ", ".join(all_preferences)
        parts.append(preference_text)

        # POI content 형식으로 1회만 추가 (임베딩 매칭용)
        for pref in all_preferences:
            mapped_type = food_type_mapping.get(pref, pref)
            parts.append(f"세계음식 > {mapped_type}")
    return ". ".join(parts)
