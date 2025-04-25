from torch import nn

class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, labels=None):
        # Embedding Layer
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, input_dim]
        embedded = self.dropout(embedded)

        # GRU Layer
        gru_out, _ = self.gru(embedded)  # gru_out: [batch_size, seq_len, hidden_dim]
        gru_out = self.dropout(gru_out)

        # Fully Connected Layer for each timestep
        logits = self.fc(gru_out)  # Shape: [batch_size, seq_len, output_dim]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
            # Reshape logits and labels for loss calculation
            logits_view = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, output_dim]
            labels_view = labels.view(-1)  # [batch_size * seq_len]
            loss = loss_fct(logits_view, labels_view)

        return logits, loss
