import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    print("Got it! Now to recognize it...")
    
    try:
        print("You said: " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("I could not understand the audio")
