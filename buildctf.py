import cntk as C
import numpy as np


def convert(filename):
    sentences = []
    vocab = set()
    with open(filename) as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            words = sentence.split()
            for word in words:
                vocab.add(word)
            sentences.append((words, int(label)))
    indexes = {w: i for (i, w) in enumerate(vocab)}
    with open("test.ctf", "w") as f:
        for i in range(len(sentences)):
            s = np.zeros(shape=(len(sentences[i][0]), len(vocab)))
            for j in range(len(sentences[i][0])):
                s[j, indexes[sentences[i][0][j]]] = 1
            print(s)

            s = np.array([[str(indexes[w])+":1"] for w in sentences[i][0]])
            label = np.array([[sentences[i][1]]])
            mapping = {"S0": s, "S1": label}
            print(C.io.sequence_to_cntk_text_format(i, mapping), file=f)
    return len(vocab)


l = convert("test.txt")
streamDefs = C.io.StreamDefs(
    sentence=C.io.StreamDef(
        field='S0', shape=l, is_sparse=True),
    label=C.io.StreamDef(
        field='S1', shape=1)
)
train_reader = C.io.MinibatchSource(
    C.io.CTFDeserializer("test.ctf",
                         streamDefs),
    randomize=True,
    max_sweeps=C.io.INFINITELY_REPEAT)
print(train_reader.next_minibatch(1))
