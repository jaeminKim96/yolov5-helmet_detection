from pydub import AudioSegment

sound = AudioSegment.from_mp3("audio2.mp3")
sound.export("audio2.wav", format="wav")
