import pandas
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def delete_http(s: str):
    s = re.sub(r"(</.*>)", "", s)
    s = re.sub(r"(<.*>)", "", s)
    return s

def solve():
    vectorizer = TfidfVectorizer()

    train = df[0:int(df.shape[0] * 0.8)]
    test = df[int(df.shape[0] * 0.8):]

    text_rate = []
    bin_rate = []

    for index, row in train.iterrows():
        if row["Score"] > min_rate:
            bin_rate.append(1)
        else:
            bin_rate.append(0)
        text_rate.append(delete_http(row["Body"]))

    fit_train = vectorizer.fit_transform(text_rate)

    classifier = LogisticRegression()
    classifier.fit(fit_train, bin_rate)

    test_text_rate = []
    test_bin_rate = []
    for index, row in test.iterrows():
        test_text_rate.append(delete_http(row["Body"]))
        if row["Score"] > min_rate:
            test_bin_rate.append(1)
        else:
            test_bin_rate.append(0)
    fit_test = vectorizer.transform(test_text_rate)
    test_predict = classifier.predict(fit_test)
    cnt = 0
    for i in range(0, len(test_predict)):
        if test_bin_rate[i] == test_predict[i]:
            cnt += 1
    print("Predicted values : ", cnt / len(test_bin_rate), sep=" ")
    return cnt / len(test_bin_rate)



path = "Answers.csv"
min_rate = 3
df = pandas.read_csv(path)
print(solve())
