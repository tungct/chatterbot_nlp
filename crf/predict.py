import pycrfsuite
from pyvi.pyvi import ViTokenizer, ViPosTagger

# with open('test.txt', 'r') as f:
#     tf = f.read().splitlines()
# l = []
# ar = []
# for i in range(len(tf)):
#     if tf[i] != "":
#         a = tf[i].split(' ')
#         l.append(tuple(a))
# ar.append(l)
# print(ar)

with open('predict.txt', 'r') as f:
    tf = f.read()
    text = ViPosTagger.postagging(ViTokenizer.tokenize(tf))
    test = []
    ar = []
    for i in range(len(text[0])):
        l = []
        l.append(text[0][i])
        l.append(text[1][i])
        ar.append(tuple(l))
    test.append(ar)


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

test_sents = test

X_test = [sent2features(s) for s in test_sents]


tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
i = 12
print(test_sents)
print(y_pred)

pred = []
for i in range(len(test_sents[0])):
    k = test_sents[0][i][0]
    v = y_pred[0][i]
    kv = []
    kv.append(k)
    kv.append(v)
    pred.append(tuple(kv))
print(pred)