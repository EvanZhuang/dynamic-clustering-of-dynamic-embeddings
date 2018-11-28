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
    return punc_free

df = pd.read_csv("~/PycharmProjects/dynamic-clustering-of-dynamic-embeddings/dat/un-general-debates.csv")
trained_topics = []
for i in range(10):
    with open('/home/yufan/PycharmProjects/dynamic-clustering-of-dynamic-embeddings/src/topic_'+str(i)+'.txt') as f:
        lines = f.read().splitlines()
        trained_topics.append(lines[0:10])

content = df['text'].tolist()
doc_clean = [clean(doc).split() for i, doc in enumerate(content)]

dictionary = corpora.Dictionary(doc_clean)
dictionary.save_as_text('./dict_lda.txt')
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

cv2 = gensim.models.CoherenceModel(topics=trained_topics, texts=doc_clean, dictionary=dictionary,
                                      coherence='c_v', window_size=110)
umass2 = gensim.models.CoherenceModel(topics=trained_topics, corpus=doc_term_matrix, dictionary=dictionary,
                                         coherence='u_mass')
cuci2 = gensim.models.CoherenceModel(topics=trained_topics, texts=doc_clean, dictionary=dictionary,
                                        coherence='c_uci', window_size=10)
cnpmi2 = gensim.models.CoherenceModel(topics=trained_topics, texts=doc_clean, dictionary=dictionary,
                                         coherence='c_npmi', window_size=10)
print("cv coherence for DCEMB: ",cv2.get_coherence())
print("umass coherence for DCEMB: ",umass2.get_coherence())
print("cuci coherence for DCEMB: ",cuci2.get_coherence())
print("cnpmi coherence for DCEMB: ",cnpmi2.get_coherence())

Lda = gensim.models.ldamulticore.LdaMulticore

ntopic = 10
ldamodel = Lda(doc_term_matrix, num_topics=ntopic, id2word=dictionary, passes=1,
               workers=multiprocessing.cpu_count()-1, eval_every=False, chunksize=1000)
ldamodel.save("./topicModel" + str(ntopic))

df = pd.DataFrame(ldamodel.print_topics(num_topics=10, num_words=10))
df.to_csv("./topics.csv", index=False)
N = 10
cv = gensim.models.CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary,
                                      coherence='c_v', topn=N, window_size=110)
umass = gensim.models.CoherenceModel(model=ldamodel, corpus=doc_term_matrix, dictionary=dictionary,
                                         coherence='u_mass', topn=N)
cuci = gensim.models.CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary,
                                        coherence='c_uci', topn=N, window_size=10)
cnpmi = gensim.models.CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary,
                                         coherence='c_npmi', topn=N, window_size=10)
print("cv coherence for LDA: ",cv.get_coherence())
print("umass coherence for LDA: ",umass.get_coherence())
print("cuci coherence for LDA: ",cuci.get_coherence())
print("cnpmi coherence for LDA: ",cnpmi.get_coherence())
