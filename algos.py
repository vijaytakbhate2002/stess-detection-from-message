import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from urllib.parse import urlparse
from spacy import load
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import joblib
import warnings
warnings.filterwarnings('ignore')

model = joblib.load("Stress identification NLP")
vectorizer = joblib.load("vectorizer_tfidf")
lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

class StressDetector:

    def __init__(self,lemmatizer,stop_words,vectorizer,model):
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words
        self.vectorizer = vectorizer
        self.model = model

    def textPocess(self,sent):
        try:
            # brackets replacing by space
            sent = re.sub('[][)(]',' ',sent)

            # url removing
            sent = [word for word in sent.split() if not urlparse(word).scheme]
            sent = ' '.join(sent)

            # removing escap characters
            sent = re.sub(r'\@\w+','',sent)

            # removing html tags 
            sent = re.sub(re.compile("<.*?>"),'',sent)

            # getting only characters and numbers from text
            sent = re.sub("[^A-Za-z0-9]",' ',sent)

            # lower case all words
            sent = sent.lower()
            
            # strip all words from sentences
            sent = [word.strip() for word in sent.split()]
            sent = ' '.join(sent)

            # word tokenization
            tokens = word_tokenize(sent)
            
            # removing words which are in stopwords
            for word in tokens:
                if word in self.stop_words:
                    tokens.remove(word)
            
            # lemmatization
            sent = [self.lemmatizer.lemmatize(word) for word in tokens]
            sent = ' '.join(sent)
            return sent
        
        except Exception as ex:
            print(sent,"\n")
            print("Error ",ex)

    def predictor(self,text):
        result = self.textPocess(text)
        result = self.vectorizer.transform([result])
        result = self.model.predict(result)
        if result == 0:
            result = "NO TRESS DETECTED"
        else:
            result = "TRESS DETECTED"
        return result

if __name__ == "__main__":

    stressed_sents = """I have an important deadline tomorrow and I haven't even started yet! #
    I'm so overwhelmed with work, I don't know how I'm going to get everything done. #
    My boss just gave me negative feedback on my project and I don't know how to fix it. #
    I can't find my keys and I'm already running late for an important meeting. #
    My car just broke down and I have no idea how I'm going to pay for the repairs. #
    I have a big exam tomorrow and I feel completely unprepared. #
    I just realized I made a huge mistake on a project that's due in a few hours. #
    I'm trying to plan a surprise party for my friend, but everything keeps going wrong. #
    My computer just crashed and I didn't save any of my work. #
    I have a job interview in an hour and I just spilled coffee all over my shirt. #
    I can't believe how much work I have to do, it feels like I'm drowning in deadlines and responsibilities. 
    I'm trying to keep up, but every time I think I'm making progress, something else comes up and I fall even further behind. 
    I just need a break, some time to catch my breath and regroup, but I don't see how that's possible. 
    I feel like I'm being pulled in a million different directions and there's no end in sight. 
    It's overwhelming and exhausting."""

    unstressed_sents = """The sun is shining and the birds are singing outside my window. #
    I'm going to grab some lunch with my coworkers today. #
    I just finished reading a really good book. #
    My dog loves going for walks in the park. #
    I'm looking forward to spending the weekend relaxing at home. #
    I need to go grocery shopping later today. #
    I enjoy listening to music while I work. #
    My favorite food is pizza. #
    I'm learning a new language and it's been really challenging but also fun. #
    I'm planning to visit my family next month. #
    I had a really great day today. In the morning, I went for a jog and enjoyed the fresh air and sunshine. 
    Then I met up with some friends for lunch and we caught up on each other's lives. 
    Later in the afternoon, I attended a seminar on a topic that interests me and learned some new things. 
    Now I'm back home, feeling content and fulfilled. It was a simple but enjoyable day and I'm grateful for it. """

    stressed_sents = stressed_sents.split("#")
    unstressed_sents = unstressed_sents.split("#")

    stress_detector = StressDetector(lemmatizer, stop_words, vectorizer, model)

    print('*'*30,"for stressed sentences","*"*30)
    for i,sent in enumerate(stressed_sents):
        result = stress_detector.predictor(sent)
        print(i,result)
    
    print('*'*30,"for unstressed sentences","*"*30)
    for i,sent in enumerate(unstressed_sents):
        result = stress_detector.predictor(sent)
        print(i,result)

