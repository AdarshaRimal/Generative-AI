import os
import json
import requests
import base64
from dotenv import load_dotenv


class OpenRouterImageGenerator:
    def __init__(self, api_key=None, save_path="generated.png"):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY missing in .env or not provided")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.save_path = save_path
        self.model = "google/gemini-2.5-flash-image-preview:free"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(self.endpoint, headers=self.headers, data=json.dumps(payload)).json()

        choice = response.get("choices", [None])[0]
        if not choice:
            print("❌ No choices returned by the API")
            return None

        message = choice.get("message", {})

        # 1️⃣ Check if image_url exists directly
        if "image_url" in message:
            img_url = message["image_url"]["url"]
            self._save_from_url(img_url)
            return self.save_path

        # 2️⃣ Check if image (base64) exists directly
        if "image" in message:
            b64_data = message["image"]["data"]
            self._save_from_base64(b64_data)
            return self.save_path

        # 3️⃣ Check content blocks (list of dicts or strings)
        contents = message.get("content", [])
        for block in contents:
            if isinstance(block, dict):
                if block.get("type") == "image_url":
                    img_url = block["image_url"]["url"]
                    self._save_from_url(img_url)
                    return self.save_path
                elif block.get("type") == "image":
                    b64_data = block["image"]["data"]
                    self._save_from_base64(b64_data)
                    return self.save_path

        # Fallback: just print text
        text_blocks = []
        for block in contents:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(block.get("text"))
            elif isinstance(block, str):
                text_blocks.append(block)
        print("ℹ️ No image found, model returned text:\n", "\n".join(text_blocks))
        return None

    def _save_from_url(self, url):
        img_data = requests.get(url).content
        with open(self.save_path, "wb") as f:
            f.write(img_data)
        print(f"✅ Saved image from URL as {self.save_path}")

    def _save_from_base64(self, b64_data):
        img_bytes = base64.b64decode(b64_data)
        with open(self.save_path, "wb") as f:
            f.write(img_bytes)
        print(f"✅ Saved image from base64 as {self.save_path}")


# ----------------------
# Usage example:
# ----------------------
if __name__ == "__main__":
    generator = OpenRouterImageGenerator()
    img_file = generator.generate("A futuristic cityscape at sunset with flying cars")
    if img_file:
        print("Image generation completed:", img_file)
    else:
        print("No image generated, only text returned.")
