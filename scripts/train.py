import pickle, os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import RecommendationDataset
from src.models.decision_transformer import DecisionTransformer
from src.training.trainer import train_decision_transformer

# Config básica
NUM_ITEMS = 752
NUM_GROUPS = 8
CONTEXT_LEN = 20
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Cargar dataset procesado
with open("data/processed/trajectories_train.pkl", "rb") as f:
    trajectories = pickle.load(f)

train_dataset = RecommendationDataset(trajectories, context_length=CONTEXT_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo
model = DecisionTransformer(
    num_items=NUM_ITEMS,
    num_groups=NUM_GROUPS,
    context_length=CONTEXT_LEN
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Entrenar
losses = train_decision_transformer(model, train_loader, optimizer, DEVICE, num_epochs=EPOCHS)

# Guardar checkpoint
os.makedirs("results/checkpoints", exist_ok=True)
ckpt_path = "results/checkpoints/dt_netflix.pth"
torch.save(model.state_dict(), ckpt_path)
print("✅ checkpoint guardado en:", ckpt_path)
