from src.sentence_embedding.sentence_embedder import InferSentEmbedder
from src.utils import cosine

if __name__ == "__main__":
    sentence1 = "Dr.Phil has already died in 1992"
    sentence2 = "Dr.Phil, died in 1991, has later become very famous."


    model = InferSentEmbedder()
    embed1 = model.encode([sentence1])[0]
    embed2 = model.encode([sentence2])[0]

    print(cosine(embed1,embed2))
