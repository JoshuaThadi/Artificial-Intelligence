import assemblyai as aai
from elevenlabs import ElevenLabs, stream
import ollama
import constants

class JOneAIVoiceAssistant:
    def __init__(self):
        # Assign API keys properly
        aai.settings.api_key = constants.ASSEMBLYAI_API_KEY
        self.client = ElevenLabs(api_key=constants.ELEVENLABS_API_KEY)

        self.transcriber = None
        self.full_transcript = [
            {"role": "system", "content": "You are a language model called R1 created by Deepseek. Answer the question being asked in less than 300 characters."}
        ]

    def start_transcription(self):
        print("\nReal-time transcription -> ")
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=19000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=19000)
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        pass

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(transcript.text)
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_close(self):
        pass

    def on_error(self, error):
        print(f"Error: {error}")

    def generate_ai_response(self, transcript):
        self.stop_transcription()
        self.full_transcript.append({"role": "user", "content": transcript.text})
        print(f'User -> {transcript.text}')

        ollama_stream = ollama.chat(
            model="deepseek-r1:5b",
            messages=self.full_transcript,  # ✅ FIXED: No parentheses here
            stream=True,
        )

        print('JOne-J1 -> ', end="\n")
        text_buffer = ""
        full_text = ""

        for chunk in ollama_stream:
            text_buffer += chunk['message']['content']
            if text_buffer.endswith('.'):
                audio_stream = self.client.generate(
                    text=text_buffer,
                    model="eleven_turbo_v2",
                    stream=True
                )
                print(text_buffer, flush=True)
                stream(audio_stream)
                full_text += text_buffer
                text_buffer = ""

        if text_buffer:
            audio_stream = self.client.generate(
                text=text_buffer,
                model="eleven_turbo_v2",
                stream=True
            )
            print(text_buffer, flush=True)
            stream(audio_stream)
            full_text += text_buffer

        self.full_transcript.append({"role": "assistant", "content": full_text})
        self.start_transcription()

# ✅ FIXED: Correctly calling the method
ai_voice_agent = JOneAIVoiceAssistant()
ai_voice_agent.start_transcription()
