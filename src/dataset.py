from torch.utils.data import Dataset
import torch


def get_sentiment_list(emb1s, emb2s, labels):
    embeddings = []
    sentiments = []
    seen = {}

    def handle_emb(emb, label):
        seen_label = seen.get(tuple(emb), label)
        assert seen_label == label
        embeddings.append(emb)
        sentiments.append(label)
        seen[tuple(emb)] = label

    for emb1, emb2, label in zip(emb1s, emb2s, labels):
        handle_emb(emb1, 1 - label)
        handle_emb(emb2, label)

    return embeddings, sentiments


class EmbeddingPM(Dataset):
    def __init__(self, emb1, emb2, labels):
        self.embeddings, self.sentiments = get_sentiment_list(emb1, emb2, labels)

    def __len__(self):
        return len(self.sentiments)

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        sentiment = (2 * self.sentiments[index]) - 1

        return torch.Tensor(embedding), torch.Tensor([sentiment])


class Embedding01(Dataset):
    def __init__(self, emb1, emb2, labels):
        self.embeddings, self.sentiments = get_sentiment_list(emb1, emb2, labels)

    def __len__(self):
        return len(self.sentiments)

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        sentiment = self.sentiments[index]

        return torch.Tensor(embedding), torch.Tensor([sentiment])
