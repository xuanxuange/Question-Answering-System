from pathlib import Path
from src.answer_generation.tokenizer import file_to_sentence as f_to_sentences

import numpy as np
import torch
import sys

#sys.path.append('.../InferSent')
from InferSent.models import InferSent

class InferSentEmbedder:

    def __init__(self):
        print("Initializing Infersent..")
        model_version = 1
        MODEL_PATH = "encoder/infersent%s.pkl" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))

        # using fastText word vector path for the model:
        W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else '../fastText/crawl-300d-2M.vec'
        model.set_w2v_path(W2V_PATH)

        # build the vocabulary of word vectors
        model.build_vocab_k_words(K=100000)

        self.model = model
        print("Infersent initializing successful")

    def encode(self, sentences):
        embeddings = self.model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        return embeddings


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#%%



if __name__ == "__main__":
    test_file = Path.cwd().parent.parent / 'data' / 'development' / 'set1' / 'a1.txt'
    sentences = f_to_sentences(test_file)

    questions_a1 = ["Was King Djoser the first king of the Fourth Dynasty of the Old Kingdom?",
                    "Who is the King of Egypt?",
                    "When did the Old Kingdom and its power reach a zenith?",
                    "How did the scholars construct a history of the 4th-6th Dynasties of Egypt?",
                    "Why did the ancient Egyptians build ships for navigation of the sea?"
                    ]
    for q in questions_a1:
        embedder = InferSentEmbedder()
        q_embedding = embedder.encode([q])[0]
        embeddings = embedder.encode(sentences)
        max = -1
        seq = -1
        for i in range(len(embeddings)):
            distance = cosine(q_embedding, embeddings[i])
            if distance > max:
                max = distance
                seq = i
        print(cosine(q_embedding, embeddings[seq]))
        print(seq)
        print(sentences[seq])