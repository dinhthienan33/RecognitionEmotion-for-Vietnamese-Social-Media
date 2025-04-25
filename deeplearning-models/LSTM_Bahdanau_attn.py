import torch
from torch import nn

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        # Linear layers to compute alignment scores
        self.attn_encoder = nn.Linear(encoder_hidden_dim, attention_dim)
        self.attn_decoder = nn.Linear(decoder_hidden_dim, attention_dim)
        self.energy = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: Tensor of shape [batch_size, seq_len, encoder_hidden_dim]
        decoder_hidden: Tensor of shape [batch_size, decoder_hidden_dim]
        """
        # Add time dimension to decoder hidden state for broadcasting
        decoder_hidden = decoder_hidden.unsqueeze(1)  # Shape: [batch_size, 1, decoder_hidden_dim]

        # Calculate alignment scores
        encoder_features = self.attn_encoder(encoder_outputs)  # Shape: [batch_size, seq_len, attention_dim]
        decoder_features = self.attn_decoder(decoder_hidden)   # Shape: [batch_size, 1, attention_dim]
        energy = torch.tanh(encoder_features + decoder_features)  # Shape: [batch_size, seq_len, attention_dim]
        alignment_scores = self.energy(energy).squeeze(-1)  # Shape: [batch_size, seq_len]

        # Compute attention weights
        attention_weights = torch.softmax(alignment_scores, dim=-1)  # Shape: [batch_size, seq_len]

        # Compute context vector as weighted sum of encoder outputs
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)  # Shape: [batch_size, encoder_hidden_dim]

        return context_vector, attention_weights
from torch import nn
import torch

class LSTMModelBahdanau(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
        attention_dim: int = 128
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
        self.attention = BahdanauAttention(hidden_dim, hidden_dim, attention_dim)
        self.fc = nn.Linear(hidden_dim + hidden_dim, output_dim)  # Concatenate hidden state and context
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
            # For each timestep, compute attention
            decoder_hidden = lstm_out[:, t, :]  # Shape: [batch_size, hidden_dim]
            context_vector, _ = self.attention(lstm_out, decoder_hidden)

            # Concatenate context vector and hidden state
            combined = torch.cat((decoder_hidden, context_vector), dim=-1)

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

