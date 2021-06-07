from fastapi import FastAPI
from functions.deep_learning import DeepL, predict_emotion
from functions.machine_learning import MachineLearning
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

app = FastAPI()

# Import dataset
df_cleaned = pd.read_csv('data/02_processed/cleaned_emotion.csv')

random_forest = MachineLearning(df_cleaned, RandomForestClassifier(), 'text_no_punct_and_stopwords_split_lem', 'Emotion')
if os.path.isfile('models/RF_model.pkl') is False: 
    random_forest.dump_model('models/RF_model.pkl')
else :
    random_forest.load_model('models/RF_model.pkl')

print("Random Forest : Ok")

logistic_regression = MachineLearning(df_cleaned, LogisticRegression(multi_class = 'multinomial'), 'text_no_punct_and_stopwords_split_lem', 'Emotion')
if os.path.isfile('models/LR_model.pkl') is False: 
    logistic_regression.dump_model('models/LR_model.pkl')
else:
    logistic_regression.load_model('models/LR_model.pkl')

print("Regression Logistic : Ok")

naive_bayes = MachineLearning(df_cleaned, MultinomialNB(), 'text_no_punct_and_stopwords_split_lem', 'Emotion')
if os.path.isfile('models/NB_model.pkl') is False:
    naive_bayes.dump_model('models/NB_model.pkl')
else:
    naive_bayes.load_model('models/NB_model.pkl')

print("Naive Bayes: Ok")

xgboost = MachineLearning(df_cleaned, XGBClassifier(objective = 'reg:logistic'), 'text_no_punct_and_stopwords_split_lem', 'Emotion')
if os.path.isfile('models/XGB_model.pkl') is False: 
    xgboost.dump_model('models/XGB_model.pkl')
else :
    xgboost.load_model('models/XGB_model.pkl')

print("XGBoost : Ok")

deep = DeepL(df_cleaned)

@app.get("/get/df")
async def getData():
        """ 
                Return Dataframe
        """
        return {"df": df_cleaned}

@app.get("/get/{model}/Classifier")
async def getClassifier(model):
        """ 
                Get Classifier of Random Forest
        """
        if model == "RandomForest" :
                return {"classifier": random_forest.model}
        elif model == "LogisticRegression" :
                return {"classifier": logistic_regression.model}
        elif model == "NaiveBayes" :
                return {"classifier": naive_bayes.model}
        elif model == "XGBoost" :
                return {"classifier": xgboost.model}
        else :
                return {"error": "model not implemented"}

@app.get("/get/{model}/Matrix")
async def getMatrix(model):
        """ 
                Get Confusion Matrix of Random Forest
        """
        val_type = "y_test"
        val_type2 = "y_pred_r"
        if model == "RandomForest" :
                return {val_type: random_forest.y_test, val_type2: random_forest.model.predict(random_forest.X_test)}
        elif model == "LogisticRegression" :
                return {val_type: logistic_regression.y_test, val_type2: logistic_regression.model.predict(random_forest.X_test)}
        elif model == "NaiveBayes" :
                return {val_type: naive_bayes.y_test, val_type2: naive_bayes.model.predict(random_forest.X_test)}
        elif model == "XGBoost" :
                return {val_type: xgboost.y_test, val_type2: xgboost.model.predict(random_forest.X_test)}
        else :
                return {"error": "model not implemented"}

@app.get("/get/{model}/Accuracy")
async def getAccuracy(model) :
        """
                Get Accuracy
        """
        val_type = "accuracy"
        if model == "RandomForest" :
                return {val_type: random_forest.calculate_accuracy()}
        elif model == "LogisticRegression" :
                return {val_type: logistic_regression.calculate_accuracy()}
        elif model == "NaiveBayes" :
                return {val_type: naive_bayes.calculate_accuracy()}
        elif model == "XGBoost" :
                return {val_type: xgboost.calculate_accuracy()}
        else :
                return {"error": "model not implemented"}

@app.get("/get/{model}/prediction/{sentence}")
async def getPrediction(model, sentence) :
        """
                Make Prediction
        """
        val_type = "prediction"
        if model == "RandomForest" :
                return {val_type: random_forest.make_prediction(sentence)}
        elif model == "LogisticRegression" :
                return {val_type: logistic_regression.make_prediction(sentence)}
        elif model == "NaiveBayes" :
                return {val_type: naive_bayes.make_prediction(sentence)}
        elif model == "XGBoost" :
                return {val_type: xgboost.make_prediction(sentence)}
        elif model == "DeepLearning" :
                return {val_type: predict_emotion(sentence, deep)}
        else :
                return {"error": "model not implemented"}
