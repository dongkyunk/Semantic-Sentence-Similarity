import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import Semantic-Sentence-Similarity.config

def avgCos(sentences1, sentences2, embedding):
    sims = []

    for (sent1, sent2) in zip(sentences1, sentences2):

        sent1 = sent1.split()
        sent2 = sent2.split()

        sent1 = [token for token in sent1 if token in embedding]
        sent2 = [token for token in sent2 if token in embedding]

        if len(sent1) == 0 or len(sent2) == 0:
            sims.append(0)
            continue

        weights1 = None
        weights2 = None

        embedding1 = np.average(
            [embedding.wv[token] for token in sent1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average(
            [embedding.wv[token] for token in sent2], axis=0, weights=weights2).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]

        sims.append(sim)

    return sims


def wordDistance(sentences1, sentences2, embedding):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        sent1 = sent1.split()
        sent2 = sent2.split()

        sent1 = [token for token in sent1 if token in embedding]
        sent2 = [token for token in sent2 if token in embedding]

        if len(sent1) == 0 or len(sent2) == 0:
            sims.append(0)
            continue

        sims.append(-embedding.wmdistance(sent1, sent2))

    return sims


def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def sifCos(sentences1, sentences2, freqs=freq, a=0.001, embedding):
    total_freq = sum(freqs.values())
    embeddings = []
    # SIF requires us to first collect all sentence embeddings and then perform
    # common component analysis.
    for (sent1, sent2) in zip(sentences1, sentences2):
        sent1 = sent1.split()
        sent2 = sent2.split()

        sent1 = [token for token in sent1 if token in embedding]
        sent2 = [token for token in sent2 if token in embedding]

        weights1 = [a/(a+freqs.get(token)/total_freq) for token in sent1]
        weights2 = [a/(a+freqs.get(token)/total_freq) for token in sent2]

        embedding1 = np.average([w2v_model.wv[token]
                                 for token in sent1], axis=0, weights=weights1)
        embedding2 = np.average([w2v_model.wv[token]
                                 for token in sent2], axis=0, weights=weights2)

        embeddings.append(embedding1)
        embeddings.append(embedding2)

    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx*2].reshape(1, -1),
                              embeddings[idx*2+1].reshape(1, -1))[0][0]
            for idx in range(int(len(embeddings)/2))]

    return sims


@print_if_complete
def feature_extraction(xtrain1, xtrain2, xvalid1, xvalid2, test):
    if config.method == "avgCos":
        sims_train = avgCos(xtrain1, xtrain2)
        sims_valid = avgCos(xvalid1, xvalid2)
        sims_test = avgCos(test["sentence1"], test["sentence2"])
    elif config.method == "wordDis":
        sims_train = wordDistance(xtrain1, xtrain2)
        sims_valid = wordDistance(xvalid1, xvalid2)
        sims_test = wordDistance(test["sentence1"], test["sentence2"])
    else:
        sims_train = avgCos(xtrain1, xtrain2)
        sims_valid = avgCos(xvalid1, xvalid2)
        sims_test = avgCos(test["sentence1"], test["sentence2"])

    sims_train = np.array(sims_train).reshape(len(sims_train), 1)
    sims_valid = np.array(sims_valid).reshape(len(sims_valid), 1)
    sims_test = np.array(sims_test).reshape(len(sims_test), 1)

    return sims_train, sims_valid, sims_test
