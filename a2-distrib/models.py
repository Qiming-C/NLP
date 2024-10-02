# models.py

import random
from collections import defaultdict

import nltk
import torch
import torch.nn as nn
from torch import optim

from sentiment_data import *


class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, embedding_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings))
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        averaged = torch.mean(embedded, dim=1).float()
        hidden = self.relu(self.fc1(averaged))
        output = self.fc2(hidden)
        return self.softmax(output)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, word_embeddings: WordEmbeddings):
        self.word_indexer = word_embeddings.word_indexer
        self.model = DeepAveragingNetwork(word_embeddings.vectors, word_embeddings.get_embedding_length(), 100, 2)
        self.unk_index = self.word_indexer.index_of("UNK")
        self.correction_cache = {}
        self.vocab = set(self.word_indexer.objs_to_ints.keys())
        self.vocab_by_prefix = defaultdict(list)
        for word in self.vocab:
            if len(word) >= 3:
                self.vocab_by_prefix[word[:3]].append(word)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:

        if has_typos:
            ex_words = [self.correct_spelling(word.lower()) for word in ex_words]

        word_indices = [self.word_indexer.index_of(word.lower()) if self.word_indexer.index_of(
            word.lower()) != -1 else self.unk_index for word in ex_words]

        word_tensor = torch.tensor([word_indices])
        probability = self.model.forward(word_tensor)
        return torch.argmax(probability, dim=1).item()

    def correct_spelling(self, word: str) -> str:
        if word in self.correction_cache:
            return self.correction_cache[word]

        if word in self.vocab or len(word) < 3:
            return word

        prefix = word[:3]
        candidates = self.vocab_by_prefix[prefix]

        if not candidates:
            return word

        best_word = word
        min_distance = float('inf')

        for candidate in candidates:

            distance = nltk.edit_distance(word[3:], candidate[3:])
            # update best word if distance is smaller
            if distance < min_distance:
                min_distance = distance
                best_word = candidate
                # Exit, perfect match found
                if distance == 0:
                    break

        self.correction_cache[word] = best_word
        return best_word


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings,
                                 train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    classifier = NeuralSentimentClassifier(word_embeddings)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters(), lr=learning_rate)
    # trainning loop
    for epoch in range(num_epochs):
        classifier.model.train()
        total_loss = 0.0
        random.shuffle(train_exs)

        # Batch the data
        for i in range(0, len(train_exs), batch_size):
            batch = train_exs[i:i + batch_size]

            # Prepare batch data
            word_indices = []
            for ex in batch:
                sentence_indices = []
                for word in ex.words:
                    index = word_embeddings.word_indexer.index_of(word)
                    if index != -1:
                        sentence_indices.append(index)
                    else:
                        sentence_indices.append(classifier.unk_index)
                word_indices.append(sentence_indices)

            # Pad sequences to the same length
            max_length = max(len(seq) for seq in word_indices)
            word_indices = [seq + [0] * (max_length - len(seq)) for seq in word_indices]

            word_tensor = torch.tensor(word_indices)
            labels = [ex.label for ex in batch]
            label_tensor = torch.tensor(labels, dtype=torch.long)
            optimizer.zero_grad()

            # Forward pass
            probs = classifier.model.forward(word_tensor)

            # Compute loss
            loss = loss_function(probs, label_tensor)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Calculate average loss
        avg_loss = total_loss / len(train_exs)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return classifier
