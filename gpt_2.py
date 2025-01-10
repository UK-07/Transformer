# -*- coding: utf-8 -*-
"""Implements a GPT model trained on Tiny Shakespeare data."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import file
import torch
from torch import nn
from torch.nn import functional as F


# Hyperparameters.
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Batch size")
_BLOCK_SIZE = flags.DEFINE_integer("block_size", 256, "Block size")
_MAX_ITERS = flags.DEFINE_integer("max_iters", 5000, "Max iterations")
_EVAL_INTERVAL = flags.DEFINE_integer("eval_interval", 500, "Eval interval")
_EVAL_ITERS = flags.DEFINE_integer("eval_iters", 200, "Eval iterations")
_EPOCHS = flags.DEFINE_integer("epochs", 5, "Epochs")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
_GAMMA = flags.DEFINE_float("gamma", 0.7, "Gamma")
_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path", None, "Path to the checkpoint."
)
_WORKDIR = flags.DEFINE_string("workdir", None, "Output directory.")
_TB_SUMMARY_LOGGING_DIR = flags.DEFINE_string(
    "tb_summary_logging_dir",
    default=os.environ.get("TB_SUMMARY_LOGGING_DIR", None),
    help="TensorBoard summary logging directory.",
)
_SEED = flags.DEFINE_integer("seed", 1337, "Seed")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_PATH = "~/gpt/data/input.txt"
_TRAIN_SPLIT = 0.9
_EMBEDDING_DIM = 384  # 384 / 6 head = 64
_LAYERS = 6
_NUM_HEADS = 6
_DROPOUT = 0.2

# Reproducibility.
torch.manual_seed(42)


class InputText:
  """Implements the vocabulary for the input text."""

  def __init__(self, text):
    self.text = text
    self.chars = sorted(list(set(text)))
    self.vocab_size = len(self.chars)

    # Tokenization strategy: single character tokens.
    self.stoi = {ch: i for i, ch in enumerate(self.chars)}
    self.itos = {i: ch for i, ch in enumerate(self.chars)}

    # Encode input in torch.Tensor
    data = torch.tensor(self.encode(text), dtype=torch.long)
    n = int(_TRAIN_SPLIT * len(data))
    self.train_data = data[:n]
    self.validation_data = data[n:]

  def encode(self, s):
    """Encodes a string into a list of token indices."""
    return [self.stoi[c] for c in s]

  def decode(self, l):
    """Decodes a list of token indices into a string."""
    return "".join([self.itos[i] for i in l])

  # Input batches.
  def get_batches(self, split):
    """Returns a batch of inputs x and targets y."""
    data = self.train_data if split == "train" else self.validation_data
    idx = torch.randint(len(data) - _BLOCK_SIZE.value, (_BATCH_SIZE.value,))
    x = torch.stack([data[i : i + _BLOCK_SIZE.value] for i in idx])
    y = torch.stack([data[i + 1 : i + _BLOCK_SIZE.value + 1] for i in idx])

    return x, y


@torch.no_grad()
def estimate_loss(model, input_text):
  """Compute average loss for multiple batches of Train and Validation."""
  out = {}
  model.eval()
  for split in ["train", "validation"]:
    losses = torch.zeros(_EVAL_ITERS.value)
    for k in range(_EVAL_ITERS.value):
      x, y = input_text.get_batches(split)
      logits, loss = model(x, y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


class Head(nn.Module):
  """Implements the model self-attention head."""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(_EMBEDDING_DIM, head_size)
    self.query = nn.Linear(_EMBEDDING_DIM, head_size)
    self.value = nn.Linear(_EMBEDDING_DIM, head_size)
    # A buffer in PyTorch is a tensor that doesn't need gradients. This is
    # separate from a parameter.
    self.register_buffer(
        "tril", torch.tril(torch.ones((_BLOCK_SIZE.value, _BLOCK_SIZE.value)))
    )
    self.dropout = nn.Dropout(_DROPOUT)

  def forward(self, x):
    """Runs one forward pass of self-attention on the given context."""
    B, T, C = x.shape  # pylint: disable=invalid-name. Batch x _BLOCK_SIZE x _EMBEDDING_DIM.
    key = self.key(x)  # (Batch x _BLOCK_SIZE x head_size)
    query = self.query(x)  # (Batch x _BLOCK_SIZE x head_size)
    value = self.value(x)  # (Batch x _BLOCK_SIZE x head_size)

    # (Batch x _BLOCK_SIZE x _BLOCK_SIZE)
    # key.trasnpose(-2, -1) - Transpose B x T x C to B x C x T.
    weights = query @ key.transpose(-2, -1) * C**-0.5
    # Tril: _BLOCK_SIZE x _BLOCK_SIZE
    # Limit tril to the size of the current context (last batch may not be full)
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    # Convert weights to probabilities. Dim = -1 ensures probabilities acroos
    # the _block dimension.
    weights = torch.softmax(weights, dim=-1)
    weights = self.dropout(weights)
    # Sum over the value based on the weights corresponding to Q/K affinity.
    token_embeddings = weights @ value
    return token_embeddings


class MultiHeadAttention(nn.Module):
  """Implements the multi-head self-attention model."""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = [Head(head_size) for _ in range(num_heads)]
    self.linear_transform = nn.Linear(_EMBEDDING_DIM, _EMBEDDING_DIM)
    self.dropout = nn.Dropout(_DROPOUT)

  def forward(self, x):
    """Runs one forward pass of self-attention on all heads."""
    logits = torch.cat([head(x) for head in self.heads], dim=-1)
    logits = self.linear_transform(logits)
    logits = self.dropout(logits)
    return logits


class FeedForward(nn.Module):
  """Implements the feed-forward network."""

  def __init__(self, embedding_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(embedding_dim, 4 * embedding_dim),
        nn.ReLU(),
        nn.Linear(4 * embedding_dim, embedding_dim),
        nn.Dropout(_DROPOUT),
    )

  def forward(self, x):
    """Runs one forward pass of the feed-forward network."""
    return self.net(x)


class Block(nn.Module):
  """Transformer block Multi-Head Atttention and Feed-Forward network."""

  def __init__(self, embedding_size, num_heads):
    super().__init__()
    head_size = embedding_size // num_heads
    self.sa_head = MultiHeadAttention(num_heads, head_size)
    # So far, the tokens have had a chance to communicate with each other via
    # self-attention. Now, they can "process" this information in the
    # feed-forward network and use it to predict the next token.
    self.ff_head = FeedForward(embedding_size)
    # Normalization happens at a per-token position level, not over multiple
    # positions across the time dimension.
    self.ln1 = nn.LayerNorm(embedding_size)
    self.ln2 = nn.LayerNorm(embedding_size)

  def forward(self, x):
    """Runs one forward pass of the transformer block."""
    # Adding residual skip-connections.
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ff_head(self.ln2(x))
    return x


class BigramLanguageModel(nn.Module):
  """Implements a BigramLanguageModel as template for GPT later on."""

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, _EMBEDDING_DIM)
    self.pos_embedding_table = nn.Embedding(_BLOCK_SIZE.value, _EMBEDDING_DIM)
    self.blocks = nn.Sequential(
        *[Block(_EMBEDDING_DIM, _NUM_HEADS) for _ in range(_LAYERS)],
    )
    self.final_ln = nn.LayerNorm(_EMBEDDING_DIM)
    self.lm_head = nn.Linear(_EMBEDDING_DIM, vocab_size)

  def forward(self, idx, targets=None):
    """Runs one forward pass of the model on given indices.

    Has B batches, each batch has T tokens (_BLOCK_SIZE length) and at each
    token position, the output will have C channels (voacb size - # of possible
    output values.) The output is a probability distribution over all possible
    values.

    Args:
      idx: (B, T) - Current token indices.
      targets: (B, T) - Target token indices.

    Returns:
      logits: (B, T, C) - Logits for each token position.
      loss: (B, T) - Loss for each token position.
    """
    B, T = idx.shape
    tok_embeddings = self.token_embedding_table(idx)  # (B, T, _EMBEDDING_DIM)
    # (T, _EMBEDDING_DIM)
    pos_embeddings = self.pos_embedding_table(torch.arange(T, device=_DEVICE))
    embeddings = tok_embeddings + pos_embeddings  # (B, T, _EMBEDDING_DIM)
    embeddings = self.blocks(embeddings)
    embeddings = self.final_ln(embeddings)
    logits = self.lm_head(embeddings)  # (B, T, vocab_size)

    if targets is None:
      # Inference / Evaluation mode.
      loss = None
    else:
      B, T, C = logits.shape
      # "Unroll" the blocks in each batch to merge all together.
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)  # Target will not have multiple channels.
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    """Generate new text of `max_new_tokens` length.

    Starts with a new line character and proceeds probabilistically.

    Args:
      idx: (B, T) - Current token indices.
      max_new_tokens: int - Number of new tokens to generate.

    Returns:
      idx: (B, T+max_new_tokens) - Updated token indices.
    """
    for _ in range(max_new_tokens):
      # Crop to the last _BLOCK_SIZE tokens.
      idx_cropped = idx[:, -_BLOCK_SIZE.value:]
      # Default function for class is forward().
      logits, loss = self(idx_cropped)
      # Focus on last timestep since Bigram model.
      logits = logits[:, -1, :]
      # Still (B, C) shape.
      probs = F.softmax(logits, dim=-1)
      # (B, 1) after sampling.
      idx_next = torch.multinomial(probs, num_samples=1)
      # (B, T+1) - Add new token to list.
      idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

  with file.Open(_PATH, "r") as f:
    text = f.read()

  input_text = InputText(text)

  model = BigramLanguageModel(input_text.vocab_size)
  m = model.to(_DEVICE)

  optimizer = torch.optim.AdamW(model.parameters(), lr=_LEARNING_RATE.value)

  for it in range(_MAX_ITERS.value):
    if it % _EVAL_INTERVAL.value == 0 or it == _MAX_ITERS.value - 1:
      loss = estimate_loss(model, input_text)
      print(
          f"Iteration: {it}\n"
          f"Training Loss: {loss['train']:.4f}\n"
          f"Validation Loss: {loss['validation']:.4f}"
      )

    xb, yb = input_text.get_batches("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  context = torch.zeros((1, 1), dtype=torch.long, device=_DEVICE)
  print(
      input_text.decode(m.generate(context, max_new_tokens=10000)[0].tolist())
  )


if __name__ == "__main__":
  app.run(main)
