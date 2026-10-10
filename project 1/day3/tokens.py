import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"
role="user"
# // 3 prompts
prompt1 = "Hi!"
prompt2 = "Explain time travel in Detail but under 100 words"
prompt3 = "Write a 1000 word essay on Machine learning"

prompts=[prompt1,prompt2,prompt3]
for prompt in prompts:
    message={
    "role": role,
    "content": prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages, max_tokens=5000)
    usage=response.usage
    print(f"Prompt: {prompt} -->your tokens: {usage.prompt_tokens} completion_tokens: {usage.completion_tokens} total tokens: {usage.total_tokens}  Finish Reason: {response.choices[0].finish_reason}")


# Ei code-e 3 ta prompt ekta list-er moddhe rakha hoyeche.
# For loop list theke ekta ekta prompt niye process kore.
# Prottek prompt-er jonno alada API request send hoy.

# Example 1: "Hi!"
# Eta simple greeting. Input ebong output token sadharonoto kom hobe.

# Example 2: "Explain time travel in detail but under 100 words"
# Ekhane 100 words-er moddhe explanation dite bola hoyeche.
# Eta prompt-er instruction, hard limit noy.

# Example 3: "Write a 1000 word essay on Machine learning"
# Ekhane long essay chaowa hoyeche.
# Tai sadharonoto output token beshi lagbe.

# max_tokens=5000 mane maximum 5000 output token allow kora.
# Er mane prottek response-e 5000 token use hobe na.

# prompt_tokens = AI-ke pathano input-er token count.
# completion_tokens = AI-er generated response-er token count.
# total_tokens = Input token + Output token.

# Token ar word ek noy.
# Ekta word ek ba ekadhik token hote pare.
# Tai word count dekhe exact token count bola jay na.

# Usage prottek API request-er alada hisab dey.
# Tin-ti prompt-er total token jante hole tin-ti request-er
# prompt_tokens, completion_tokens ebong total_tokens jog korte hobe.

# Mone rakhbe:
# For loop = Prottek prompt ekta kore process kore.
# API request = Model-er kache prompt pathay.
# Response = Model-er generated answer.
# Usage = Request-e koto token use hoyeche tar hisab.



# prompt="Do you know Padho with Pratyush"
# # message me role and content
# message={
#     "role": role,
#     "content": prompt
# }

# messages=[message]

# response=client.chat.completions.create(model=model, messages=messages)