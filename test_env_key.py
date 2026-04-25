import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载 .env 环境变量
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

print(f"[*] 正在使用的 API_BASE_URL: {base_url}")
print(f"[*] 正在使用的 API_KEY (前5位): {api_key[:5] if api_key else 'None'}...")

if not api_key:
    print("[!] 错误：环境变量中未找到 OPENAI_API_KEY，请检查 .env 文件。")
    exit(1)

# 2. 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 3. 尝试进行简单的生成调用
print("\n[*] 正在向大模型发送网络请求，请稍候...")
try:
    response = client.chat.completions.create(
        model="deepseek-chat",  # 或是你通常使用的模型名称（如 gpt-3.5-turbo 等）
        messages=[
            {"role": "user", "content": "你好，请回复'连接成功'这四个字。"}
        ],
        max_tokens=20
    )
    print("\n[+] 恭喜！API 连通性测试通过！")
    print(f"[+] 模型回复: {response.choices[0].message.content}")

except Exception as e:
    print("\n[-] 测试失败，出现以下异常：")
    print("-" * 40)
    import traceback
    traceback.print_exc()
    print("-" * 40)
    print("[!] 请根据以上报错重点检查：网络是否连通被墙、BASE_URL是否填写争取、或者你的 API Key 是否有效。")
