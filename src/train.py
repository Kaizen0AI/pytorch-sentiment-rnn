import torch
import torch.nn as nn
import torch.optim as optim
from src.model import RNNClassifier
from src.data import create_dataloaders
from src.constants import MAX_VOCAB_SIZE, NUM_EPOCHS
from tqdm import tqdm
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.helper_funcs import EarlyStoppingAndCheckpoint


def train(device, model, criterion, optimizer, train_loader, val_loader, scheduler, tracker, num_epochs=NUM_EPOCHS):

    def validate(model, val_loader, criterion, device):
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_logits = model(X_val_batch)
                val_loss = criterion(val_logits.squeeze(1), y_val_batch)

                predictions = (torch.sigmoid(val_logits) > 0.5).float() 
                correct_predictions += (predictions.squeeze(1) == y_val_batch).sum().item()
                total_samples += y_val_batch.size(0)
                running_val_loss += val_loss.item()

        epoch_val_acc = (correct_predictions / total_samples) * 100
        epoch_val_loss = running_val_loss / len(val_loader)
        return epoch_val_loss, epoch_val_acc

    history = {
        "loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=True)

        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits.squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            running_loss += current_loss
            loop.set_postfix(loss=current_loss)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_loss = running_loss / len(train_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch +1}/{NUM_EPOCHS}] | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        scheduler.step(val_loss)

        tracker(val_loss, model, optimizer, epoch)
                
        history["loss"].append(float(epoch_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history_path = os.getenv("HISTORY_PATH", "history/history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f)
        #print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if tracker.early_stop:
            print("\n[Early Stopping Triggered] Stopping training early.")
            break
    return history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(vocab_size = MAX_VOCAB_SIZE, embed_dim = 128, hidden_dim= 128, output_dim= 1 ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train_loader, val_loader = create_dataloaders()

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    tracker = EarlyStoppingAndCheckpoint(patience=5, filepath="checkpoints/best_model.pt")

    history = train(device=device, 
                    model=model, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    scheduler=scheduler,
                    tracker=tracker)
