from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
try:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Test"}])
    print("API key works!")
except Exception as e:
    print(f"Error: {e}")