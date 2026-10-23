
import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"


# Structure the output using Pydantic
class Ticket(BaseModel):
    name: str
    email: str
    issue: str


schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract information from the customer ticket.

Return only a valid JSON object matching this schema:
{schema}

Use only the fields defined in the schema.
If information is missing, use an empty string.
"""

message_system = {
    "role": "system",
    "content": system_prompt
}

text = "Hello myself Arpan i am from kolkata i bought a iphone but its not working my email is abc@gmail.com"

prompt = f"""
Extract the customer's information from this ticket:

{text}
"""

messages = [
    message_system,
    {
        "role": "user",
        "content": prompt
    }
]

# Get a JSON response from Groq
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.2,
    response_format=response_format
)

# Read the model's response
rawjson = response.choices[0].message.content

if not rawjson:
    raise ValueError("Model returned an empty response")

print("Raw JSON response:")
print(rawjson)

# Convert JSON string into a Python dictionary
datafile = json.loads(rawjson)

# Validate the data using Pydantic
ticket = Ticket(**datafile)

print("\nExtracted Information:")
print("Name:", ticket.name)
print("Email:", ticket.email)
print("Issue:", ticket.issue)
