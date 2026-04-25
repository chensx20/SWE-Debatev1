import os
import logging
from dotenv import load_dotenv
load_dotenv(".env")

# Add localization directory to path
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'localization'))

from entity_localization_pipeline import EntityLocalizationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_connection():
    pipeline = EntityLocalizationPipeline()
    print(f"[*] API Base: {pipeline.client.base_url}")
    print(f"[*] Model: {pipeline.model_name}")
    
    messages = [{"role": "user", "content": "Hello, respond with 'Success'."}]
    try:
        print("[*] Testing _call_llm_simple...")
        response = pipeline._call_llm_simple(messages)
        print(f"[+] Response: {response}")
        return True
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
