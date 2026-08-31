import torch
import os

class EarlyStoppingAndCheckpoint:
    def __init__(self, patience=5, min_delta=1e-4, filepath=os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")):
        self.patience = patience
        self.min_delta = min_delta
        self.filepath = filepath
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, epoch):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, optimizer, val_loss, epoch)
            print(f"  --> Validation loss improved. Checkpoint saved to '{self.filepath}'")
        else: 
            self.counter += 1
            print(f"  --> No improvement in val_loss. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, optimizer, val_loss, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }

        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(checkpoint, self.filepath)