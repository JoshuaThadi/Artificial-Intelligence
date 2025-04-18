import os
import speech_recognition as sr
import pyaudio
import pywhatkit  # type: ignore

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