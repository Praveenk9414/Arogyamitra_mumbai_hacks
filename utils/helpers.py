# utils/helpers.py
import re

def normalize_proc_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    synonyms = {
        "appendicitis": "appendectomy",
        "appendix removal": "appendectomy",
        "appendix": "appendectomy",
        "meniscus tear": "partial_medial_meniscus_tear",
        "medial meniscus tear": "partial_medial_meniscus_tear",
        "cataract": "cataract_surgery",
        "cataract surgery": "cataract_surgery",
    }
    for k, v in synonyms.items():
        if k in s:
            return v
    return re.sub(r"\s+", "_", s)
