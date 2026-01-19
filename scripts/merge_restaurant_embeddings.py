#!/usr/bin/env python3
"""
Merge embedding_restaurant_tmp.json and embedding_restaurant.json

- Uses content from embedding_restaurant_tmp.json as the new content field
- Moves the original content from embedding_restaurant.json to a new 'tags' field
- All other fields remain unchanged from embedding_restaurant.json
"""

import json
from pathlib import Path


def merge_restaurant_embeddings():
    # File paths
    base_dir = Path(__file__).parent.parent / "data" / "embedding_json"
    tmp_file = base_dir / "embedding_restaurant_tmp.json"
    orig_file = base_dir / "embedding_restaurant.json"
    output_file = base_dir / "embedding_restaurant_merged.json"

    # Load both files
    print(f"Loading {tmp_file}...")
    with open(tmp_file, "r", encoding="utf-8") as f:
        tmp_data = json.load(f)

    print(f"Loading {orig_file}...")
    with open(orig_file, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    # Verify same length
    if len(tmp_data) != len(orig_data):
        raise ValueError(
            f"File lengths don't match: tmp={len(tmp_data)}, orig={len(orig_data)}"
        )

    print(f"\nProcessing {len(orig_data)} entries...")

    # Create merged data
    merged_data = []
    for tmp_entry, orig_entry in zip(tmp_data, orig_data):
        # Verify same place_id
        if tmp_entry["place_id"] != orig_entry["place_id"]:
            raise ValueError(
                f"place_id mismatch: tmp={tmp_entry['place_id']}, "
                f"orig={orig_entry['place_id']}"
            )

        # Start with original entry
        merged_entry = orig_entry.copy()

        # Move original content to tags field
        merged_entry["tags"] = orig_entry["content"]

        # Replace content with tmp content
        merged_entry["content"] = tmp_entry["content"]

        merged_data.append(merged_entry)

    # Save merged data
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # Show sample output
    print("\n=== Sample Output (first entry) ===")
    print(f"place_id: {merged_data[0]['place_id']}")
    print(f"name: {merged_data[0]['name']}")
    print(f"tags: {merged_data[0]['tags']}")
    print(f"content: {merged_data[0]['content']}")

    print(f"\n✓ Successfully merged {len(merged_data)} entries")
    print(f"✓ Output saved to: {output_file}")
    print(f"\nTo replace the original file, run:")
    print(f"  cp {output_file} {orig_file}")


if __name__ == "__main__":
    merge_restaurant_embeddings()
