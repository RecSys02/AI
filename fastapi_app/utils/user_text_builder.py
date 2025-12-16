# utils/user_text_builder.py

def build_user_profile_text(user) -> str:
    parts = []

    if user.city:
        parts.append(f"여행 도시: {user.city}")

    if user.companion_type:
        parts.append("동행 유형: " + ", ".join(user.companion_type))

    if user.preferred_themes:
        parts.append("선호 테마: " + ", ".join(user.preferred_themes))

    if user.preferred_moods:
        parts.append("선호 분위기: " + ", ".join(user.preferred_moods))

    if user.activity_level:
        parts.append(f"활동 강도: {user.activity_level}")

    if user.budget:
        parts.append(f"예산 수준: {user.budget}")

    if user.avoid:
        parts.append("피하고 싶은 요소: " + ", ".join(user.avoid))

    return ". ".join(parts)
