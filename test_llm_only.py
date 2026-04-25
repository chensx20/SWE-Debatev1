import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")

def test_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    
    print(f"[*] API Base: {base_url}")
    print(f"[*] API Key (first 5): {api_key[:5]}...")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        print("[*] Sending request...")
        response = client.chat.completions.create(
            model="gpt-5.4-xhigh",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"[+] Success: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

if __name__ == "__main__":
    test_llm()

