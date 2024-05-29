import pandas as pd
from flask import Flask, request, render_template
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask("__name__")

@app.route("/")
def loadPage():
    return render_template('index.html', query1="")

@app.route("/", methods=['POST'])
def LanguagePrediction():
    df = pd.read_csv("Language Detection.csv")

    inputQuery1 = request.form['query1']
    X = df["Text"]
    y = df["Language"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    data_list = []
    for text in X:
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        data_list.append(text)

    cv = CountVectorizer()
    X = cv.fit_transform(data_list).toarray()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    ac = accuracy_score(y_test, y_pred)*100
    print("Accuracy is:", ac)

    def predict(text):
        ac = accuracy_score(y_test, y_pred)
        print("Accuracy is:", ac)
        x = cv.transform([text]).toarray()
        lang = model.predict(x)
        lang = le.inverse_transform(lang)
        print("The language is in", lang[0])
        return lang[0]
    

    output=predict(inputQuery1)
    return render_template('index.html',output1=output,output2=ac, query1=inputQuery1)

if __name__ == "__main__":
    app.run()
