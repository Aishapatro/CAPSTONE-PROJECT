import speech_recognition as sr
import pyttsx3 as pt
import pywhatkit as pk

# Initialize speech recognition
listening = sr.Recognizer()

# Initialize text-to-speech
engine = pt.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def hear():
    cmd = ""

    try:
        with sr.Microphone() as mic:

            print("Adjusting microphone...")
            listening.adjust_for_ambient_noise(mic, duration=1)

            print("Listening...")

            voice = listening.listen(mic, timeout=10, phrase_time_limit=10)

            print("Recognizing speech...")

            cmd = listening.recognize_google(voice)
            cmd = cmd.lower()

            print("You said:", cmd)

            if "aisha" in cmd:
                cmd = cmd.replace("aisha", "").strip()

    except sr.WaitTimeoutError:
        print("No speech detected.")

    except sr.UnknownValueError:
        print("Sorry, I could not understand.")

    except sr.RequestError as e:
        print("Google API Error:", repr(e))

    except Exception as e:
        print("Error Type:", type(e).__name__)
        print("Error Details:", repr(e))

    return cmd


def run():
    cmd = hear()

    if "play" in cmd:
        song = cmd.replace("play", "").strip()

        if song:
            speak("Playing " + song)
            pk.playonyt(song)

    else:
        print("Command not recognized.")


run()
