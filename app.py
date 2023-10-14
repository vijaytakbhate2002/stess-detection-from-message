import speech_recognition as sr
from algos import StressDetector
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

print("Model uploading...")
model = joblib.load("Stress identification NLP")
vectorizer = joblib.load("vectorizer_tfidf")
lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

print("System ready")
r = sr.Recognizer()
class StressPredictor:
    def __init__(self,lemmatizer,stop_words,vectorizer,model):
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words
        self.vectorizer = vectorizer
        self.model = model

    def speech_input(self):
        with sr.Microphone() as source:
            print("Speak something...")
            audio = r.listen(source)
            return audio

    def stress_predictor(self):
        """ Recognize speech using Google Speech Recognition """
        SD = StressDetector(self.lemmatizer, self.stop_words, self.vectorizer, self.model)
        audio = self.speech_input()
        try:
            text = r.recognize_google(audio)
            print(text)
            return SD.predictor(text)
        except sr.UnknownValueError:
            return "Sorry, I could not understand what you said."
        except sr.RequestError as e:
            return f"Request error: {e}"

if __name__ == "__main__":
    detector = StressPredictor(lemmatizer,stop_words,vectorizer,model)
    while True:
        print(detector.stress_predictor())