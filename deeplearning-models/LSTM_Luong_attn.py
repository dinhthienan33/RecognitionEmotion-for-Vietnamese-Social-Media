import torch
from torch import nn
import torch
from torch import nn

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, method: str = 'dot'):
        """
        Luong Attention Mechanism
        Args:
            hidden_dim: The dimensionality of the hidden state vectors.
            method: The scoring method, either 'dot', 'general', or 'concat'.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'general':
            self.attn = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Parameter(torch.rand(hidden_dim))

    def score(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention scores.
        Args:
            decoder_hidden: The current decoder hidden state, shape [batch_size, hidden_dim].
            encoder_outputs: Encoder outputs for all timesteps, shape [batch_size, seq_len, hidden_dim].
        Returns:
            Attention scores, shape [batch_size, seq_len].
        """
        if self.method == 'dot':
            # Dot product of decoder hidden state and encoder outputs
            return torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # [batch_size, seq_len]
        elif self.method == 'general':
            # Transform encoder outputs before dot product
            transformed = self.attn(encoder_outputs)  # [batch_size, seq_len, hidden_dim]
            return torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)  # [batch_size, seq_len]
        elif self.method == 'concat':
            # Concatenate decoder hidden state and encoder outputs
            seq_len = encoder_outputs.size(1)
            decoder_hidden_exp = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
            concat = torch.cat((decoder_hidden_exp, encoder_outputs), dim=2)  # [batch_size, seq_len, hidden_dim*2]
            scores = torch.tanh(self.attn(concat))  # [batch_size, seq_len, hidden_dim]
            return torch.bmm(scores, self.v.unsqueeze(0).unsqueeze(2).expand_as(scores)).squeeze(2)  # [batch_size, seq_len]

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention-weighted context vector.
        Args:
            decoder_hidden: The current decoder hidden state, shape [batch_size, hidden_dim].
            encoder_outputs: Encoder outputs for all timesteps, shape [batch_size, seq_len, hidden_dim].
        Returns:
            Context vector, shape [batch_size, hidden_dim].
            Attention weights, shape [batch_size, seq_len].
        """
        # Compute attention scores
        scores = self.score(decoder_hidden, encoder_outputs)  # [batch_size, seq_len]

        # Normalize scores to probabilities (attention weights)
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]

        # Compute context vector as the weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, hidden_dim]

        return context, attn_weights

class LSTMModelLuong(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
        attn_method: str = 'dot'
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0
        )
        self.attention = LuongAttention(hidden_dim, method=attn_method)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Concatenate hidden state and context
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, labels=None):
        # Embedding Layer
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, input_dim]
        embedded = self.dropout(embedded)

        # LSTM Layer
        lstm_out, _ = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)

        # Attention and Context Vector
        logits = []
        for t in range(lstm_out.size(1)):
            # Decoder hidden state for timestep t
            decoder_hidden = lstm_out[:, t, :]  # Shape: [batch_size, hidden_dim]

            # Compute attention context vector
            context_vector, _ = self.attention(decoder_hidden, lstm_out)  # Shape: [batch_size, hidden_dim]

            # Concatenate context vector and decoder hidden state
            combined = torch.cat((decoder_hidden, context_vector), dim=-1)  # Shape: [batch_size, hidden_dim * 2]

            # Fully Connected Layer
            logit = self.fc(combined)  # Shape: [batch_size, output_dim]
            logits.append(logit)

        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, seq_len, output_dim]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
            # Reshape logits and labels for loss calculation
            logits_view = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, output_dim]
            labels_view = labels.view(-1)  # [batch_size * seq_len]
            loss = loss_fct(logits_view, labels_view)

        return logits, loss
