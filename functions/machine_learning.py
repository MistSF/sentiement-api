# Import global librairies
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import NLP librairies
from nltk.corpus import stopwords


# Import Machine Learning librairies
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pickle
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# Import Deep Learning librairies
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import sequence, text




# Create MachineLearning class
class MachineLearning:
    def __init__(self, df ,model, X_col, y_col):
        self.df = df
        self.model = model
        self.X_col = X_col
        self.y_col = y_col
        
        self.vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        X = self.vectorizer.fit_transform(self.df[self.X_col].apply(lambda x: np.str_(x)))
        tfidfconverter = TfidfTransformer()
        self.X = tfidfconverter.fit_transform(X).toarray()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.df[self.y_col], test_size=0.2, random_state=0)
        self.model.fit(self.X_train, self.y_train)    
        
    def dump_model(self, filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.classifier, file)
        
    def load_model(self, filepath):
        with open(filepath, 'rb') as file:  
            self.model = pickle.load(file)
        return self.classifier

    def calculate_accuracy(self):
        y_pred_r = self.model.predict(self.X_test)
        return metrics.accuracy_score(self.y_test, y_pred_r)
    
    def make_prediction(self, sentence):
        return (self.classifier.predict(self.vectorizer.transform([sentence])))
   
        
# Create DeepLearning class
class DeepLearning:
    def __init__(self, df, X_col, y_col):
        self.df = df
        self.X = X_col
        self.y = y_col
        self.tokenization()
        self.word_embedding()
        self.split_dataset()
    
    def tokenization(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=400000,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ")
        self.max_len = 70

        self.tokenizer.fit_on_texts(self.df[self.X])
        word_index = self.tokenizer.word_index

        # wi_len = len(word_index)

    def word_embedding(self):
        self.X = self.df[self.X].values
        self.X = self.tokenizer.texts_to_sequences(self.X)
        self.X = sequence.pad_sequences(self.X, maxlen=self.max_len)
    
    def split_dataset(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=46, test_size=0.3)
    
    def load_model(self, load_model, load_weigth):
        # load json and create model
        json_file = open(load_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(load_weigth)
        st.write("Loaded model from disk")
    
    def evaluate_model(self):
        # evaluate loaded model on test data
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        score = self.loaded_model.evaluate(self.x_train, self.y_train, verbose=0)
        print("%s: %.2f%%" % (self.loaded_model.metrics_names[1], score[1]*100))
