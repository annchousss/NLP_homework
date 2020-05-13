import nltk
import pandas
import pymorphy2
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords


# get normalized review function
def get_normalized_review(text):
    analyzer = pymorphy2.MorphAnalyzer()
    normalized_words = []
    tokens = nltk.word_tokenize(text)
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '–']
    for token in tokens:
        if token in puncto: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    # added some stopwords which were received after the 1st run
    stopword_set = set(stopwords.words("russian"))
    stopword_set.add('фильм')
    stopword_set.add('свои')
    stopword_set.add('это')
    stopword_set.add('весь')
    stopword_set.add('который')
    normalized_words = [w for w in normalized_words if not w in stopword_set]
    return normalized_words


# filter reviews by its title
def filter_by_titles(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


def main():
    dictionary = corpora.Dictionary(df['text'])
    corpus = [dictionary.doc2bow(text) for text in df['text']]

    # define numbers of topics and words
    topics_number_const = 10
    words_number_const = 15
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics_number_const, id2word=dictionary, passes=15)

    for index, topic in lda.show_topics(num_topics=topics_number_const, formatted=False, num_words=words_number_const):
        print('Topic: {} \nWords: {}'.format(index, [word[0] for word in topic]))

    cm = CoherenceModel(model=lda, texts=df['text'], dictionary=dictionary)
    coherence = cm.get_coherence()
    print(coherence)


# all reviews
df = pandas.read_csv("filmreviews.csv", encoding="utf-8")
df['text'] = df['text'].map(lambda x: get_normalized_review(x))
