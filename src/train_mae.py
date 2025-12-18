import torch

def train_mae(mae, dataloader, optimizer, mse_loss, device, epochs=100):
    for epoch in range(epochs):
        mae.train()
        total_loss, total_patches = 0, 0

        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()

            recon, idx_mask = mae(X)
            target = patchify(X)

            recon_m = recon[:, idx_mask]
            target_m = target[:, idx_mask]

            loss = mse_loss(recon_m, target_m)
            loss.backward()
            optimizer.step()

            B, Pm = recon_m.shape[:2]
            total_loss += loss.item() * B * Pm
            total_patches += B * Pm

        print(f"[MAE] Epoch {epoch+1} | loss={total_loss/total_patches:.6f}")
