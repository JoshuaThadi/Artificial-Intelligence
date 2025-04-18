import google.generativeai as genai
import datetime

# Configure the API
genai.configure(api_key="GEMINI_API_KEY")

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Start a chat session
chat_session = model.start_chat(history=[])

print("Real-time Chatbot Started (Type 'exit' to quit)\n")

while True:
    user_input = input(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] You -> ")
    
    if user_input.lower() == "exit":
        print("Bot Terminated!")
        break
    
    response = chat_session.send_message(user_input)  
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Gemini -> {response.text}")
