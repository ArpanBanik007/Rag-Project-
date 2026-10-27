import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)

model="openai/gpt-oss-120b"

prompt = "Who is Virat Kohli?"

messages = [
    {
        "role": "user",
        "content": prompt
    }
]


response = client.chat.completions.create(
    model=model,
    messages=messages
)

# print(response.choices[0].message.content)


# Stream code -----------------

stream=client.chat.completions.create(model=model, messages=messages,stream=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
       print(content, end="", flush=True)