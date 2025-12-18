import numpy as np
from sklearn.metrics import f1_score

def finetune_classifier(model, train_loader, val_loader,
                        optimizer, criterion, device, epochs=100):

    for epoch in range(epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        train_acc = correct / total

        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv).argmax(1)
                all_y.append(yv.cpu())
                all_p.append(pv.cpu())

        f1 = f1_score(
            np.concatenate(all_y),
            np.concatenate(all_p),
            average="binary"
        )

        print(f"[FT] Epoch {epoch+1} | acc={train_acc:.3f} | f1={f1:.3f}")
