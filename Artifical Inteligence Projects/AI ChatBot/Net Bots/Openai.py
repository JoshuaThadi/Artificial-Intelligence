''' The OpenAI api key requires plans to initiate (It is paid)'''

from openai import OpenAI

openai = OpenAI(api_key = 'openai_key',)
conversation = []

def get_gpt_response(user_input):
    message = {"role": "user", "content": user_input}
    conversation.append(message)
    response = openai.chat.completions.create(messages = conversation,model = "gpt-3.5-turbo")
    conversation.append(response.choice[0].message)
    return response.choices[0].message.content

def chat():
    while True:
        user_input = input("You -> ")
        if user_input.lower() == 'exit':
            print("Gpt -> Shutting Down!")
            break
        
        response = get_gpt_response(user_input)
        print(f"Gpt -> {response}")
        
if __name__ == "__main__":
    chat()
    
'''import openai

openai.api_key = "openai_key"  # Replace with your actual API key

completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "write an essay on penguins"}]
    )
print(completion.choices[0].message.content)'''