#!/usr/bin/env python3
"""
Normalize restaurant tags to 5 categories: 한식, 중식, 양식, 일식, 세계음식

Rules:
- "한국음식 > ..." -> 한식
- "세계음식 > 양식" -> 양식
- "세계음식 > 일식" -> 일식
- "세계음식 > 중식" -> 중식
- Everything else -> 세계음식
"""

import json
from pathlib import Path
from collections import Counter


def normalize_tag(tag: str) -> str:
    """Normalize a tag to one of the 5 categories."""
    if not tag or not isinstance(tag, str):
        return "세계음식"

    tag = tag.strip()

    # 한국음식 > ... -> 한식
    if tag.startswith("한국음식 >"):
        return "한식"

    # 세계음식 > 양식 -> 양식
    if "세계음식 > 양식" in tag or tag == "양식":
        return "양식"

    # 세계음식 > 일식 -> 일식
    if "세계음식 > 일식" in tag or tag == "일식":
        return "일식"

    # 세계음식 > 중식 -> 중식
    if "세계음식 > 중식" in tag or tag == "중식":
        return "중식"

    # Everything else -> 세계음식
    return "세계음식"


def normalize_restaurant_tags():
    # File paths
    base_dir = Path(__file__).parent.parent / "data" / "embedding_json"
    input_file = base_dir / "embedding_restaurant.json"
    output_file = base_dir / "embedding_restaurant_normalized.json"

    # Load data
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries...\n")

    # Track tag transformations
    original_tags = Counter()
    normalized_tags = Counter()
    transformations = {}

    # Normalize tags
    for entry in data:
        original_tag = entry.get("tags", "")
        normalized_tag = normalize_tag(original_tag)

        entry["tags"] = normalized_tag

        # Track statistics
        original_tags[original_tag] += 1
        normalized_tags[normalized_tag] += 1

        if original_tag not in transformations:
            transformations[original_tag] = normalized_tag

    # Save normalized data
    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n=== Normalization Statistics ===")
    print(f"\nOriginal unique tags: {len(original_tags)}")
    print(f"Normalized unique tags: {len(normalized_tags)}")

    print("\n=== Normalized Tag Distribution ===")
    for tag, count in sorted(normalized_tags.items(), key=lambda x: -x[1]):
        percentage = count / len(data) * 100
        print(f"{tag}: {count} ({percentage:.1f}%)")

    print("\n=== Sample Tag Transformations ===")
    shown = set()
    for original, normalized in sorted(transformations.items()):
        if normalized not in shown or len(shown) < 20:
            print(f"{original:40s} -> {normalized}")
            shown.add(normalized)

    print(f"\n✓ Successfully normalized {len(data)} entries")
    print(f"✓ Output saved to: {output_file}")
    print(f"\nTo replace the original file, run:")
    print(f'  copy "{output_file}" "{input_file}"')


if __name__ == "__main__":
    normalize_restaurant_tags()
