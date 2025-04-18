import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import time

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def talk(text):
    """ Makes Jarvis speak and prints response """
    print(f"Jarvis: {text}") 
    engine.say(text)
    engine.runAndWait()

def take_command():
    """ Listens only when speech is detected and captures full command """
    with sr.Microphone() as source:
        print("Listening...")
        listener.adjust_for_ambient_noise(source, duration=1)

        try:
            voice = listener.listen(source, timeout=5, phrase_time_limit=8)  # Allows up to 8 sec of speech
            command = listener.recognize_google(voice).lower()

            if 'Jarvis' in command:
                command = command.replace('Jarvis', '').strip()
            
            print(f"You: {command}")
            return command
        
        except sr.UnknownValueError:
            print("Jarivs: Sorry, I didn't catch that.")
            return ""  
        except sr.RequestError:
            talk("Speech recognition service is unavailable.")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""

def search_wikipedia(query):
    """ Searches Wikipedia and handles exceptions """
    try:
        info = wikipedia.summary(query, sentences=2)  # Increased to 2 sentences for better info
        talk(info)
    except wikipedia.exceptions.DisambiguationError as e:
        talk(f"Multiple results found. Please be more specific. Example: {e.options[:3]}")
    except wikipedia.exceptions.PageError:
        talk("Sorry, I couldn't find anything on that topic.")
    except Exception as e:
        talk(f"An error occurred: {e}")

def run_jarvis():
    """ Runs only when a valid command is given """
    command = take_command()
    if command:  # Only process if there's a command
        if 'play' in command:
            video_audio = command.replace('play', '').strip()
            talk(f"Playing {video_audio}")
            pywhatkit.playonyt(video_audio)
        elif 'time' in command:
            time_now = datetime.datetime.now().strftime('%I:%M %p')
            talk(f"Current time is {time_now}")
        elif 'search' in command or 'who is' in command or 'who are' in command:
            search_query = command.replace('search', '').replace('who is', '').replace('who are', '').strip()
            search_wikipedia(search_query)
        elif 'joke' in command:
            talk(pyjokes.get_joke())
        elif 'exit' in command or 'stop' in command:
            talk("Goodbye!")
            return False  # Exit the loop when the user says "exit" or "stop"
        else:
            talk("I did not understand. Please try again.")
    time.sleep(2)  # Wait before listening again
    return True

while True:
    if not run_jarvis():  # Break the loop if 'exit' or 'stop' is detected
        break
