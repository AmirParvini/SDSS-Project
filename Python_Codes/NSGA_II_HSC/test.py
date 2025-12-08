from httpx._transports import base
from openai import OpenAI
import json
import os
import ai_config

client_type = "groq"

if client_type == "openai":
    # ایجاد یک نمونه از کلاینت با کلید API خود
    client = OpenAI(base_url= ai_config.BASE_URL, api_key=ai_config.OPENROUTER_API_KEY)
    MODEL = ai_config.AI_MODEL
    def save_response_id(response_id, filename="response_ids.json"):
        """
        Saves a response ID to a JSON file.
        If the file exists, it appends the new ID to the existing list.
        If the file does not exist, it creates a new file with the ID.
        """
        data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Handle case where file is empty or malformed
                    data = []
        
        data.append(response_id)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load_last_response_id(filename="response_ids.json"):
        """
        Loads the last saved response ID from a JSON file.
        Returns None if the file does not exist or is empty/malformed.
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                    if data:
                        return data[-1] # Return the last ID in the list
                except json.JSONDecodeError:
                    pass
        return None

    # Check for existing response ID
    previous_id = load_last_response_id()

    if previous_id:
        print(f"Using existing response ID: {previous_id}")
        res2 = client.responses.create(
            model=MODEL,
            input="what is my name?",
            previous_response_id=previous_id,
            store=True
        )
    else:
        print("No existing response ID found. Creating a new one.")
        res1 = client.responses.create(
            model=MODEL,
            input="my name is amir. just say 'hi'",
            store=True
        )
        print(res1)
        save_response_id(res1.id) # Save the new ID
        res2 = client.responses.create(
            model=MODEL,
            input="what is my name?",
            previous_response_id=res1.id,
            store=True
        )

    print(res2.output_text)
    
if client_type == "groq":
    from groq import Groq
    api_key = "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"
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "hi"
            }
        ]
    )
    print(completion.choices[0].message.content)