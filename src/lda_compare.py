import gensim
import pandas as pd
import os, re, string
import multiprocessing
from gensim import corpora
from sklearn.model_selection import train_test_split

exclude = set(string.punctuation)
exclude.update(set(string.digits))
def clean(doc):

    doc = doc.lower()

    punc_free = ''.join(ch for ch in doc if ch not in exclude)

    # counter = collections.Counter(normalized.split())
    # normalized = ' '.join(ch for ch in normalized.split() if counter[ch] > 1)
    # finder = BigramCollocationFinder.from_words(normalized.split(), window_size=3)
    # bigram_measures = nltk.collocations.BigramAssocMeasures()
    # bigrams = finder.score_ngrams(bigram_measures.pmi)
    # print(normalized)
    # print(bigrams)
    # return

    return punc_free
target_path = "~/PycharmProjects/dynamic-clustering-of-dynamic-embeddings/dat/"
df = pd.read_csv("~/PycharmProjects/dynamic-clustering-of-dynamic-embeddings/dat/un-general-debates.csv")
content = df['text'].tolist()
doc_clean = [clean(doc).split() for i, doc in enumerate(content)]
print(doc_clean[0])
dictionary = corpora.Dictionary(doc_clean)
dictionary.filter_extremes(keep_n=10000)
dictionary.save_as_text('./dict_post35.txt')
doc_train, doc_test = train_test_split(doc_clean, test_size=1000, random_state=0)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_train]
doc_test_matrix = [dictionary.doc2bow(doc) for doc in doc_test]
Lda = gensim.models.ldamulticore.LdaMulticore
print("MISSILE LAUNCHED")

ntopic = 10
ldamodel = Lda(doc_term_matrix, num_topics=ntopic, id2word=dictionary, passes=1,
               workers=2, eval_every=False, chunksize=1000)
ldamodel.save("./topicModel" + str(ntopic))

df = pd.DataFrame(ldamodel.print_topics(num_topics=10, num_words=10))
df.to_csv("./topics.csv", index=False)
N = 10
cv = gensim.models.CoherenceModel(model=ldamodel, texts=doc_test, dictionary=dictionary,
                                      coherence='c_v', topn=N, window_size=110)

print(cv.get_coherence())

