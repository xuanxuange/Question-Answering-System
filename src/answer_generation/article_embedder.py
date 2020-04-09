from src.answer_generation.tokenizer import file_to_sentence
from src.sentence_embedding.sentence_embedder import InferSentEmbedder
from src.utils import cosine

class ArticleEmbedder:

    def __init__(self, fp):
        self.sentences = file_to_sentence(fp)
        print("initialize article embedding for file {}, number of sentences: {}".format(fp, len(self.sentences)))
        self.infersent_embedder = InferSentEmbedder()
        self.infersent_embedded_sententes = self.infersent_embedder.encode(self.sentences)
        print("finish embedding sentences using Infersent, number of sentences {}".format(len(self.infersent_embedded_sententes)))


    def get_most_relevant_sentence_infersent(self, sentence):
        embedded_base_sentence = self.infersent_embedder.encode([sentence])[0]
        max = -1
        seq = -1
        # cos distance, 1 for most relavant, -1 for most irrelavant
        for i in range(len(self.infersent_embedded_sententes)):
            distance = cosine(embedded_base_sentence, self.infersent_embedded_sententes[i])
            if distance > max:
                max = distance
                seq = i
        return self.sentences[seq]

