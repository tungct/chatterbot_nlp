import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import pickle
import pycrfsuite
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


def write(data, outfile):
    f = open(outfile, "w+b")
    pickle.dump(data, f)
    f.close()

with open('tag_vtv.txt', 'r') as f:
    trf = f.read().splitlines()
ls = []
arr = []

for i in range(len(trf)):
    if trf[i] != "":
        a = trf[i].split(' ')
        # print(i)
        # if len(a)> 3:
        #     print(a[0] + " " + a[1] + " " + a[2])
        #     print(i)
        if len(a) != 3:
            print(a[0] + " " + a[1])
        # del a[1]
        # if a[0] == '' or a[1] == '' or a[2] == '':
        #     print(a)
        ls.append(tuple(a))
arr.append(ls)

# print(nltk.corpus.conll2002.fileids())
train_sents = arr
#
with open('predict.txt', 'r') as f:
    tf = f.read().splitlines()
l = []
ar = []
for i in range(len(tf)):
    if tf[i] != "":
        a = tf[i].split(' ')
        # del a[1]
        # if a[0] == '' or a[1] == '' or a[2] == '':
        #     print(a)
        l.append(tuple(a))
ar.append(l)

test_sents = ar

# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

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


def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

X_train = [extract_features(doc) for doc in train_sents]
y_train = [get_labels(doc) for doc in train_sents]
#
X_test = [extract_features(s) for s in test_sents]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')
#
# write(crf, "trains.file")

y_pred = crf.predict(test_sents)

# print(y_pred)
print("XXX")
print(crf.predict_single(test_sents[0][0]))
import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
# print(y_pred)
print("XXX")
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
i = 12
# print(y_pred)
# for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
#     print("%s (%s)" % (y, x))
# print(crf.predict_marginals(X_test))
# print(metrics.flat_f1_score(y_test, y_pred,
#                       average='weighted', labels=labels))
# # group B and I results
# sorted_labels = sorted(
#     labels,
#     key=lambda name: (name[1:], name[0])
# )
# print(metrics.flat_classification_report(
#     y_test, y_pred, labels=sorted_labels, digits=3
# ))
