# -------- Instruction to create a voice assistant

# required libraries
pip install pywhatkit
pip install pyaudio
pip install speech_recognition

# step 1 --->

def get_audio():
    recorder = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recorder.listen(source)

    text = recorder.recognize_google(audio)
    print(f"You -> {text}")
    return text

if __name__ == "__main__":
    get_audio()


# step 2 --->
def get_audio():
    recorder = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recorder.listen(source)

    text = recorder.recognize_google(audio)
    print(f"You -> {text}")
    return text


#text = get_audio()
pywhatkit.playonyt("NetworkChuck")

# step 3 ->
def get_audio():
    recorder = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recorder.listen(source)

    text = recorder.recognize_google(audio)
    print(f"You -> {text}")
    return text


text = get_audio()
pywhatkit.playonyt(text)

# step 4 ->
def get_audio():
    recorder = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recorder.listen(source)

    text = recorder.recognize_google(audio)
    print(f"You -> {text}")
    return text


text = get_audio()
if "youtube" in text.lower():
    pywhatkit.playonyt(f'Siri -> {text}')
else:
    pywhatkit.search(f'Siri -> {text}')