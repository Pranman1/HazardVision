import json
from anthropic import Anthropic

def summarize_logs():
    with open("backend/hazard_log.json") as f:
        logs = json.load(f)

    text = json.dumps(logs[-10:], indent=2)
    prompt = f"Summarize these workplace hazard logs:\n{text}"

    client = Anthropic(api_key="sk-your-key-here")
    response = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return {"summary": response.content[0].text}
