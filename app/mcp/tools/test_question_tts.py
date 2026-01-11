# Install TTS (run this once in your terminal, then continue with Python)
# !pip install TTS

from TTS.api import TTS

# Download and load the English TTS model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# Generate audio for a sample interview question
question = "Hello, welcome to your AI interview. Please tell me about yourself."
tts.tts_to_file(text=question, file_path="question.wav")
print("Audio file question.wav generated!")
