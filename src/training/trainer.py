import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_decision_transformer(model, train_loader, optimizer, device, num_epochs=10):
    """
    Entrena el Decision Transformer con cross-entropy.
    targets trae el próximo item real en la secuencia.
    """
    model.train()
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)
            groups = batch["groups"].to(device)
            targets = batch["targets"].to(device)

            logits = model(states, actions, rtg, timesteps, groups)

            loss = F.cross_entropy(
                logits.reshape(-1, model.num_items),
                targets.reshape(-1),
                ignore_index=-1
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1}: loss promedio = {avg_loss:.4f}")

    return losses
