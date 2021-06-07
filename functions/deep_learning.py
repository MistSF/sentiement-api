from operator import mod
import numpy as np
import pandas as pd 
import tensorflow as tf
import nltk
import seaborn as sns
import re
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
label_encoder = preprocessing.LabelEncoder()

#Example
def predict_emotion(stri, model):
    review = re.sub('[^a-zA-Z]', ' ', stri)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review,10000)] 
    embed = pad_sequences(onehot_repr,padding='pre',maxlen=35)
    predicti = model.predict(embed)
    return label_encoder.classes_[np.argmax(predicti)]

def DeepL(df) :
    df=df.dropna()
    X=df.drop('Emotion',axis=1)
    y=df['Emotion']

    ### Vocabulary size
    voc_size=10000

    messages=X.copy()
    messages.reset_index(inplace=True)

    nltk.download('stopwords')

    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['Text'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    onehot_repr=[one_hot(words,voc_size)for words in corpus] 

    #Finding max words
    l = 0
    for x in corpus:
        l = max(l,len(x.split(' ')))

    sent_length=35
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

    # Creating model
    embedding_vector_features=300
    model=Sequential()
    model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
    model.summary()

    X_final=np.array(embedded_docs)
    y = label_encoder.fit_transform(y)
    y_final=np.array(y)

    X_final.shape,y_final.shape

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=21)

    model_save = ModelCheckpoint('models/weights.h5', save_best_only = True, save_weights_only = True, monitor = 'val_loss', 
                                mode = 'min', verbose = 1)
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=15,batch_size=128,callbacks = [model_save])

    model.load_weights('models/weights.h5')
    y_pred=model.predict_classes(X_test)
    print(y_pred)
    st.write(accuracy_score(y_test,y_pred))
    st.text(classification_report(y_test, y_pred, digits=5))
    print('Confusion Matrix')
    plt.figure(figsize=(15,10))
    sns.heatmap(confusion_matrix(y_test, y_pred), fmt = 'd', annot = True)
    plt.xticks(rotation=50)
    plt.show()
    st.pyplot()
    # print(sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt="d"))
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(le_name_mapping)

    return model