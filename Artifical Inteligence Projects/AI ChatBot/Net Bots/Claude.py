''' Claude platform is paid to initiate the program'''
import anthropic

# Initialize the Anthropic client with the correct API key
client = anthropic.Anthropic(api_key='claude_key')

conversation = []

while True:
    user_message = input("You: ")
    if user_message.lower() == "exit":
        print("Chat ended. Goodbye!")
        break
    
    conversation.append({'role': 'user', 'content': user_message})

    response = client.messages.create(
        model='claude-3-5-sonnet-20241022',  # Ensure this is the correct model name
        max_tokens=1024,
        messages=conversation
    )

    bot_message = response.content  # Correct way to access the response content
    print(f"Claude: {bot_message}")

    conversation.append({'role': 'assistant', 'content': user_message})