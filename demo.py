from algos import StressDetector
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from urllib.parse import urlparse
from spacy import load
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import app
import warnings
warnings.filterwarnings('ignore')

model = joblib.load("Stress identification NLP")
vectorizer = joblib.load("vectorizer_tfidf")
lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))
model = joblib.load('Stress identification NLP')

sents = [
"I'm running out of time, and the deadline is looming over me!",
"I can't handle all the pressure and expectations placed on me!",
"Every decision I make feels like a potential disaster waiting to happen.",
"I feel like I'm constantly playing catch-up, and I can never get ahead.",
"The workload keeps piling up, and there's no end in sight.",
"I'm overwhelmed by the sheer number of responsibilities I have to juggle.",
"I'm afraid of making a mistake that could cost me my job or reputation.",
"It feels like everyone is counting on me, and I can't afford to let them down.",
"The office politics and constant competition are driving me to the edge.",
 "No matter how hard I work, I always feel inadequate and behind everyone else."
]

stress_detector = app.StressDetector(lemmatizer,stop_words,vectorizer,model)

for sent in sents:
    print(stress_detector.predictor(sent))