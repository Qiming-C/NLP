# models.py

import math
import numpy as np
import torch
from torch import nn, optim
from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index
        self.seq_length = model.seq_length


    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            # Pad the context with spaces on the left side if it's shorter than seq_length - 1
            padded_context = context.rjust(self.seq_length - 1)
            # Take the last (seq_length - 1) characters of the padded context
            context = padded_context[-(self.seq_length - 1):]
            
            context_indices = [self.vocab_index.index_of(char) for char in context]
            context_tensor = torch.tensor(context_indices).unsqueeze(0)
            
            logits = self.model(context_tensor)
            last_token_logits = logits[0, -1, :]
            
            log_probs = torch.log_softmax(last_token_logits, dim=0)

        return log_probs.numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob_sum = 0.0
        current_context = context

        for char in next_chars:
            char_log_probs = self.get_next_char_log_probs(current_context)
            char_index = self.vocab_index.index_of(char)
            log_prob_sum += char_log_probs[char_index]
            current_context += char
            if len(current_context) >= self.seq_length - 1:
                current_context = current_context[-(self.seq_length - 1):]

        return log_prob_sum


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, feedforward_dim, seq_length, dropout=0.1):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions=seq_length)
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            feedforward_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        seq_len = src.size(1)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = self.dropout(src)

        output = self.transformer(src, mask=mask)
        output = self.dropout(output)

        return self.output_layer(output)
      
def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    vocab_size = len(vocab_index)
    d_model = 256
    nhead = 8
    num_layers = 4
    feedforward_dim = 256
    batch_size = 64
    seq_length = 50
    num_epochs = 20
    learning_rate = 0.001


    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=nhead,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        seq_length=seq_length
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_data = torch.tensor([vocab_index.index_of(char) for char in train_text])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data) - seq_length, batch_size * seq_length):
            batch_end = i + batch_size * seq_length
            if batch_end > len(train_data) - seq_length:
                break

            batch = train_data[i:batch_end].reshape(batch_size, seq_length)
            targets = train_data[i + 1:batch_end + 1].reshape(batch_size, seq_length)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
    
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    
    model.eval()
    return NeuralLanguageModel(model, vocab_index)