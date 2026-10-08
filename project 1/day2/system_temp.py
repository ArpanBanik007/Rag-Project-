import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"

prompt = " suggest a name for my tech company"

# messages_system = {
#     "role": "system",
#     "content": "Act like my very sweet and romantic gf."
# }
messages_system = {
    "role": "system",
    "content": "Act like a very good content  manager ."
}

# testing temparature by giving both promt


# messages_system = {
#     "role": "system",
#     "content":"Act like my very rude and strict boss."
# }

messages = [
    messages_system,
    {
        "role": "user",
        "content": prompt
    }
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.1

    #when i give more temparature is close to 2 give more crative and close to 0 give like strite
  
)

print(response.choices[0].message.content)