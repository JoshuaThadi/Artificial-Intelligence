import asyncio
import datetime
import requests
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Function to get user location
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return f"{data.get('city', 'Unknown City')}, {data.get('country', 'Unknown Country')}"
    except:
        return "Unknown Location"

# Get current location & time
user_location = get_location()
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Define prompt template with real-time data
template = f"""
Answer the user's question based on the conversation history.
Current Time: {current_time}
User Location: {user_location}

Conversation History:
{{history}}

User: {{question}}
Ollama:
"""

# Initialize LLM model with streaming enabled
model = OllamaLLM(model="llama3.2", streaming=True)

# Create a structured prompt template
prompt = ChatPromptTemplate.from_template(template)

# Define processing chain
chain = prompt | model


async def handle_conversation():
    history = ""
    print(f"\nOllama: Hello! I'm Ollama your local device AI assistant. I see you're in {user_location}. Type 'exit' to stop.\n")

    while True:
        user_input = await asyncio.to_thread(input, "You -> ")

        if user_input.lower() in ["exit", "quit"]:
            print("\nOllama -> Goodbye! Have a great day!\n")
            break

        # Stream the response in real-time
        print("Ollama -> ", end="", flush=True)
        result = ""
        async for chunk in chain.astream({"history": history, "question": user_input}):
            print(chunk, end="", flush=True)
            result += chunk

        print("\n")  # Move to new line after response

        # Update conversation history
        history += f"\nUser: {user_input}\nOllama: {result}"


if __name__ == "__main__":
    asyncio.run(handle_conversation())
