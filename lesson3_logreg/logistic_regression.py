import nltk
import numpy
import pandas
import pymorphy2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


# function for getting normalized reviews
def get_normalized_review(text):
    analyzer = pymorphy2.MorphAnalyzer()
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '#', '$', '%', '&']
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in puncto: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


# function for getting a bag of words
def get_bag_of_words(full_text, vocabulary):
    normalized_texts = []
    for elem in full_text:
        normalized_texts.append(get_normalized_review(elem[1]))
    if len(vocabulary) == 0:
        vocabulary = numpy.unique(numpy.concatenate(normalized_texts))
    word_vectors = []
    for text in normalized_texts:
        bag_vector = numpy.zeros(len(vocabulary))
        for w in text:
            for i, word in enumerate(vocabulary):
                if word == w:
                    bag_vector[i] += 1
        word_vectors.append(bag_vector)
    return vocabulary, word_vectors


# all reviews
df = pandas.read_csv("filmreviews.csv", encoding="utf-8")
film_reviews = ["Престиж", "Побег из Шоушенка", "Послезавтра"]

# filter by test reviews
test_data = df[df['title'].isin(film_reviews)]
df = df[~df['title'].isin(film_reviews)]

# create bag of words
vocabulary, bag_of_words = get_bag_of_words(df.values, [])
vocabulary, test_bag_of_words = get_bag_of_words(test_data.values, vocabulary)

# train model and
reg = LogisticRegression(max_iter=10000)
reg.fit(bag_of_words, df['label'])

# apply the model to texts
predicted = reg.predict(test_bag_of_words)

# get accuracy
positive = 0
p = 0
for prediction in predicted:
    if prediction == test_data.values[p][2]:
        positive += 1
    p += 1

print("Accuracy: {}".format(positive / len(predicted)))
# get precision, recall, fscore
precision_recall_fscore = precision_recall_fscore_support(test_data['label'].values, predicted)
print(f"Precision_vector: {precision_recall_fscore[0]}")
print(f"Recall_vector: {precision_recall_fscore[1]}")
print(f"Fscore_vector: {precision_recall_fscore[2]}")

# dictionaries of negavive, neutral and positive words
dict_of_negative = dict(zip(vocabulary, reg.coef_[0]))
dict_of_neutral = dict(zip(vocabulary, reg.coef_[1]))
dict_of_positive = dict(zip(vocabulary, reg.coef_[2]))

# sort our dictionaries
sorted_negative_dictionary = {k: v for k, v in sorted(dict_of_negative.items(), key=lambda item: item[1], reverse=True)}
sorted_neutral_dictionary = {k: v for k, v in sorted(dict_of_neutral.items(), key=lambda item: item[1], reverse=True)}
sorted_positive_dictionary = {k: v for k, v in sorted(dict_of_positive.items(), key=lambda item: item[1], reverse=True)}

# write vocabularies to files
negativewords = open("negativewords.txt", "w", encoding="utf-8")
neutralwords = open("neutralwords.txt", "w", encoding="utf-8")
positivewords = open("positivewords.txt", "w", encoding="utf-8")

print(sorted_negative_dictionary, file=negativewords)
print(sorted_neutral_dictionary, file=neutralwords)
print(sorted_positive_dictionary, file=positivewords)

