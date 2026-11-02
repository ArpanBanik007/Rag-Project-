import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path


 
load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)

model="openai/gpt-oss-120b"




# step 1
knowledge_base={
    "name":"Arpan Banik",
    "age" :"The age of arpan is 22 years",
    "net worth" : "The net worth of arpan is 2000000000000"
}


# step 2 retreieval
def retrieve_info(question):
    question=question.lower()

    if "name" in question:
        return knowledge_base["name"]
    elif "age" in question:
        return knowledge_base["age"]
    elif  "net worth" in question:
        return knowledge_base[ "net worth"]
    else:
        return None


def ask_llm(question):
    context=retrieve_info(question)

    sys_prompt=f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
    system_message={
        "role": "system",
        "content": sys_prompt

    }
    message={
        "role": "user",
        "content": question
    }
    messages=[system_message, message]
    response=client.chat.completions.create(model=model, messages=messages)
    answer=response.choices[0].message.content
    return answer


question="what is Arpan Banik's age?"
print(ask_llm(question))