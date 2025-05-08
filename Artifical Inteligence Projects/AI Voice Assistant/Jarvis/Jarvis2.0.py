import pyttsx3 # for voice
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import smtplib
import os

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine_voice_category = engine.setProperty('voice', voices[0].id)
print(engine_voice_category)
print(voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning Boss! How may I assist you?")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon Boss! How may I assist you?")
    else:
        speak("Good Evening Boss! How may I assist you?")

def takeCommand():
    # takes microphone input and return string output
    re = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        re.pause_threshold = 1
        audio = re.listen(source)

    try:
        print("recognizing...")
        query = re.recognize_google(audio, language='en-in')
        print(f"User command : {query}\n")

    except Exception as e:
        print(e)
        print('Pardom me, Please repeat yourself')
        return "None"
    return query
        
contacts = {
    "josh": "thadijoshua@gmail.com",
    "john": "john.doe@example.com",
    "alice": "alice.smith@example.com"
}
        
def sendEmail(to, content):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('youremail@gmail.com', 'your-password')  # Use App Password for security
        server.sendmail('youremail@gmail.com', to, content)
        server.close()
        return True
    except Exception as e:
        print(e)
        return False
    
if __name__ == "__main__":
    wishme()
   
    while True:
        query = takeCommand().lower()
        
        if 'Internet' in query:
            speak('Searching from Wikipedia...')
            query = query.replace("search about", "")
            results = wikipedia.summary(query, sentences = 2)
            speak(f'According to wikipedia -> {results}')
            print(results)
            
        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
            print("command executed -> youtube opened")

        elif 'open google' in query:
            webbrowser.open("google.com")
            print("command executed -> google opened")
        
        elif 'open edge' in query:
            webbrowser.open("edge.com")
            print("command executed -> edge opened")
        
        elif 'open firefox' in query:
            webbrowser.open("firefox.com")
            print("command executed -> firefox opened")
            
        elif 'open opera' in query:
            webbrowser.open("opera.com")
            print("command executed -> opera opened")
            
        elif 'open stackoverflow' in query:
            webbrowser.open("stackoverflow.com")
            print("command executed -> stackoverflow opened")
            
        elif 'open chatgpt' in query:
            webbrowser.open("chatgpt.com")
            print("command executed -> chatgpt opened")
            
        elif 'play music' in query:
            music_dir = r"C:\Users\Joshua\OneDrive\Music"
            if not os.path.exists(music_dir):  
                speak("Music folder not found.")  
            else:  
                songs = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.wav', '.m4a'))]  # Filter music files

                if not songs:
                    speak("No music files found in the directory.")
                else:
                    print(f"Playing: {songs[0]}")
                    os.startfile(os.path.join(music_dir, songs[0]))  # Open first file
            print("command executed -> music playing...")
                
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H: %M: %S")
            speak(f'The current time in your location is {strTime}')
            print("command executed -> Time showed")
            
        elif 'open code' in query:
            vspath = r"C:\Users\Joshua\AppData\Local\Programs\Microsoft VS Code\bin\code"
            os.startfile(vspath)
            print("command executed -> vscode opened")
        
        elif 'email to' in query:
            try:
                recipient_name = query.replace('email to', '').strip().lower()
                
                if recipient_name in contacts:
                    speak("What should I say?")
                    content = takeCommand()
                    to = contacts[recipient_name]
                    if sendEmail(to, content):
                        speak(f"Email has been sent to {recipient_name}!")
                    else:
                        speak("Sorry Boss! I was unable to send the email.")
                else:
                    speak("Sorry, this contact is not in your list.")
            
            except Exception as e:
                print(e)
                speak("Sorry Boss! I am not able to send the message.")
                
        elif 'go to sleep' in query or 'exit' in query:
            speak("GoodBye Boss! Its been nice chatting with you.")
            exit()