import json
import re

def extract_json_obj(text: str):
    """Best-effort extraction of a JSON object from a model response."""
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        return json.loads(m.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    return json.loads(text)
