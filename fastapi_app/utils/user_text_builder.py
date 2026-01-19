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
    history = getattr(user, "history_places", None) or getattr(user, "visit_tourspot", None)
    if history:
        ts = [p for p in history if getattr(p, "category", None) == "tourspot"]
        if ts:
            parts.append(f"방문한 관광지 수: {len(ts)}")
    selected = getattr(user, "selected_places", None) or getattr(user, "last_selected_pois", None)
    if selected:
        last_ts = [p for p in selected if getattr(p, "category", None) == "tourspot"]
        if last_ts:
            ids = [str(p.place_id) for p in last_ts if getattr(p, "place_id", None) is not None]
            parts.append("마지막 선택 관광지: " + ", ".join(ids))
    return ". ".join(parts)


def build_cafe_text(user) -> str:
    parts = _base_profile_lines(user)
    if user.preferred_cafe_types:
        # 쉼표로 구분된 경우 분리 처리
        all_preferences = []
        for pref in user.preferred_cafe_types:
            pref_items = [p.strip() for p in pref.split(',')]
            all_preferences.extend(pref_items)

        # 원본 형식을 여러 번 반복하여 가중치 강화 (3회 반복)
        preference_text = "선호 카페 타입: " + ", ".join(all_preferences)
        parts.append(preference_text)
        parts.append(preference_text)
        parts.append(preference_text)

        # 카페 타입을 5회 반복하여 임베딩 공간에서 강한 시그널 생성
        for pref in all_preferences:
            # 각 타입을 5번 반복
            for _ in range(5):
                parts.append(f"카페 > {pref}")

        # 추가 키워드 강조
        for pref in all_preferences:
            parts.append(f"{pref} 전문점")
            parts.append(f"{pref} 카페")
            parts.append(f"{pref}만 추천")

    if user.preferred_moods:
        parts.append("카페 분위기: " + ", ".join(user.preferred_moods))
    history = getattr(user, "history_places", None) or getattr(user, "visit_cafe", None)
    if history:
        cafes = [p for p in history if getattr(p, "category", None) == "cafe"]
        if cafes:
            parts.append(f"방문한 카페 수: {len(cafes)}")
    selected = getattr(user, "selected_places", None) or getattr(user, "last_selected_pois", None)
    if selected:
        last_cafe = [p for p in selected if getattr(p, "category", None) == "cafe"]
        if last_cafe:
            ids = [str(p.place_id) for p in last_cafe if getattr(p, "place_id", None) is not None]
            parts.append("마지막 선택 카페: " + ", ".join(ids))
    return ". ".join(parts)


def build_restaurant_text(user) -> str:
    parts = _base_profile_lines(user)
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

        # 원본 형식을 여러 번 반복하여 가중치 강화 (3회 반복)
        preference_text = "선호 음식점 타입: " + ", ".join(all_preferences)
        parts.append(preference_text)
        parts.append(preference_text)
        parts.append(preference_text)

        # POI content 형식으로 추가 (임베딩 매칭 향상) - 5회 반복으로 강력한 시그널
        for pref in all_preferences:
            mapped_type = food_type_mapping.get(pref, pref)
            poi_format = f"세계음식 > {mapped_type}"
            # 각 음식 타입을 5번 반복하여 임베딩 공간에서 강한 시그널 생성
            for _ in range(5):
                parts.append(poi_format)

        # 추가 키워드 강조 (한식/양식 등을 독립적으로도 강조)
        for pref in all_preferences:
            mapped_type = food_type_mapping.get(pref, pref)
            parts.append(f"{mapped_type} 전문점")
            parts.append(f"{mapped_type} 맛집")
            parts.append(f"{mapped_type}만 추천")
    history = getattr(user, "history_places", None) or getattr(user, "visit_restaurant", None)
    if history:
        rests = [p for p in history if getattr(p, "category", None) == "restaurant"]
        if rests:
            parts.append(f"방문한 음식점 수: {len(rests)}")
    selected = getattr(user, "selected_places", None) or getattr(user, "last_selected_pois", None)
    if selected:
        last_rest = [p for p in selected if getattr(p, "category", None) == "restaurant"]
        if last_rest:
            ids = [str(p.place_id) for p in last_rest if getattr(p, "place_id", None) is not None]
            parts.append("마지막 선택 음식점: " + ", ".join(ids))
    return ". ".join(parts)
