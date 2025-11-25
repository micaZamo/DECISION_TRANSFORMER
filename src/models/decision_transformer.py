import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        num_items=752,
        num_groups=8,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        context_length=20,
        max_timestep=200,
        dropout=0.1
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Embeddings
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Head de predicción
        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items)
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, states, actions, returns_to_go, timesteps, user_groups, attention_mask=None):
        """
        Args:
            states: (B, L) IDs de items vistos (historia)
            actions: (B, L) IDs recomendados (en este TP usamos igual a states)
            returns_to_go: (B, L, 1) returns-to-go
            timesteps: (B, L) posiciones temporales
            user_groups: (B,) grupo del usuario
        Returns:
            logits: (B, L, num_items)
        """
        B, L = states.shape

        state_emb = self.item_embedding(states)              # (B, L, H)
        rtg_emb = self.rtg_embedding(returns_to_go)          # (B, L, H)
        time_emb = self.timestep_embedding(timesteps)        # (B, L, H)

        group_emb = self.group_embedding(user_groups).unsqueeze(1)   # (B, 1, H)
        group_emb = group_emb.expand(-1, L, -1)                      # (B, L, H)

        h = state_emb + rtg_emb + time_emb + group_emb
        h = self.ln(h)

        if attention_mask is None:
            attention_mask = self._generate_causal_mask(L).to(h.device)

        h = self.transformer(h, mask=attention_mask)
        logits = self.predict_item(h)

        return logits

    def _generate_causal_mask(self, L):
        """Máscara causal para que cada posición solo vea el pasado."""
        return torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1)
