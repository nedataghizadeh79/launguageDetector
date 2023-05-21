import string
import pandas as pd
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn import linear_model
from sklearn import metrics

df = pd.read_csv("Language Detection 2.csv")

def remove_pun(text):
    for pun in string.punctuation:
        text = text.replace(pun, "")
    text = text.lower()
    return text

# print(df.head())
# Text Language
# 0   Nature, in the broadest sense, is the natural...  English
# 1  "Nature" can refer to the phenomena of the phy...  English
# 2  The study of nature is a large, if not the onl...  English
# 3  Although humans are part of nature, human acti...  English
# 4  [1] The word nature is borrowed from the Old F...  English

df['Text'] = df['Text'].apply(remove_pun)

X = df.iloc[:, 0] # the sentence
Y = df.iloc[:, 1] # name of the language

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), analyzer='char')
model_pipe = pipeline.Pipeline([("vec", vec), ("clf", linear_model.LogisticRegression())])

model_pipe.fit(X_train, Y_train)


predict_val = model_pipe.predict(X_test)
print(predict_val)

metrics.accuracy_score(Y_test, predict_val)*100

metrics.confusion_matrix(Y_test, predict_val)

res = model_pipe.predict(['Hi, this is Neda!'])
print(res)