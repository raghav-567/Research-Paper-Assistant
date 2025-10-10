from dotenv import load_dotenv
import google.generativeai as genai
import os
load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key)

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
