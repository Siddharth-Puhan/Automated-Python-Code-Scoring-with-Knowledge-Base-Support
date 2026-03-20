import json
from app import build_normalized_kb, match_topic_to_kb_enhanced


def test_match_binary_search():
    """Ensure KB matching finds a Binary Search topic from the Knowledge_base."""
    normalized = build_normalized_kb()
    sample_code = (
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr)-1\n"
        "    while lo <= hi:\n"
        "        mid = (lo+hi)//2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid+1\n"
        "        else:\n"
        "            hi = mid-1\n"
        "    return -1\n"
    )

    category, topic, score, candidates, tags, description = match_topic_to_kb_enhanced(
        "Binary Search", sample_code, normalized
    )

    assert category is not None
    assert topic is not None
    assert "binary" in topic.lower() or "search" in category.lower()
