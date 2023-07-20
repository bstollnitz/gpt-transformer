# MIT License
# Copyright (c) 2023 Bea Stollnitz

import tiktoken
import torch

from transformer import make_model, subsequent_mask


def tokenize(text, batch_size):
    """Convert text to numerical tokens and repeat batch_size times."""
    encoding = tiktoken.encoding_for_model("davinci")
    token_list = encoding.encode(text)
    token_tensor = torch.tensor(token_list, dtype=torch.long)  # (input_seq_len)
    token_tensor = token_tensor.unsqueeze(0)  # (1, input_seq_len)
    token_tensor = token_tensor.repeat(batch_size,
                                       1)  # (batch_size, input_seq_len)
    return encoding, token_tensor


def limit_sequence_length(input_tokens, block_size):
    """Limit the input to at most block_size tokens."""
    input_seq_len = input_tokens.size(1)
    seq_len = min(input_seq_len, block_size)
    block_tokens = input_tokens[:, -seq_len:]  # (batch_size, seq_len)
    return block_tokens


def generate_next_token(model, tokens):
    """Use the highest probability from the Transformer model to choose the next token."""
    mask = subsequent_mask(tokens.size(1))  # (1, seq_len, seq_len)
    decoder_output = model.decode(tokens,
                                  mask)  # (batch_size, seq_len, vocab_size)
    distribution = model.generator(
        decoder_output[:, -1, :])  # (batch_size, vocab_size)
    next_token = torch.argmax(distribution, dim=1,
                              keepdim=True)  # (batch_size, 1)
    return next_token


# Define constants.
input_text = "A long time ago"
new_token_count = 10
batch_size = 1
block_size = 1024

# Tokenize the input text.
encoding, tokens = tokenize(input_text, batch_size)

# Create the model.
model = make_model(encoding.n_vocab)

# Iterate until we've generated enough new tokens.
for _ in range(new_token_count):
    block_tokens = limit_sequence_length(tokens,
                                         block_size)  # (batch_size, seq_len)
    next_token = generate_next_token(model, block_tokens)  # (batch_size, 1)
    tokens = torch.cat([tokens, next_token],
                       dim=1)  # (batch_size, input_seq_len + 1)

# Print each of the generated token sequences.
print("Output:")
print(tokens)
for row in tokens:
    print(encoding.decode(row.tolist()))
