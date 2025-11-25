import numpy as np

class PopularityRecommender:
    """
    Baseline 1: recomienda ítems según popularidad global.
    """
    def __init__(self, num_items=752):
        self.num_items = num_items
        self.item_counts = None
        self.popular_items = None

    def fit(self, train_data):
        all_items = np.concatenate([traj["items"] for traj in train_data])
        self.item_counts = np.bincount(all_items, minlength=self.num_items)
        self.popular_items = np.argsort(self.item_counts)[::-1]

    def recommend(self, user_history, k=10):
        recs = []
        seen = set(user_history)
        for item in self.popular_items:
            if item not in seen:
                recs.append(item)
                if len(recs) == k:
                    break
        return recs


class RandomRecommender:
    """
    Baseline 2: recomendaciones aleatorias.
    """
    def __init__(self, num_items=752, seed=42):
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

    def fit(self, train_data):
        pass

    def recommend(self, user_history, k=10):
        seen = set(user_history)
        candidates = [i for i in range(self.num_items) if i not in seen]
        return self.rng.choice(candidates, size=k, replace=False).tolist()

class BehaviorCloningTransformer(nn.Module):
    """
    Baseline 3: Modelo Transformer supervisado que aprende a predecir la siguiente acción
    (próxima película) únicamente a partir del historial del usuario
    """
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

        # === EMBEDDINGS (sin rtg) ===
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)

        # === TRANSFORMER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True          
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        # === HEAD PREDICTORA ===
        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items)
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, states, timesteps, user_groups, attention_mask=None):
        B, L = states.shape

        state_emb = self.item_embedding(states)
        time_emb = self.timestep_embedding(timesteps)

        group_emb = self.group_embedding(user_groups).unsqueeze(1).expand(-1, L, -1)

        # Suma de embeddings
        h = self.ln(state_emb + time_emb + group_emb)

        # === Máscara causal moderna ===
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(states.device)

        if attention_mask is not None:
            key_padding = (attention_mask == 0)
        else:
            key_padding = None

        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=key_padding)

        logits = self.predict_item(h)
        return logits