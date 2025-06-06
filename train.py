import os, time, json, copy, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import SparseAutoencoder

class SparseAutoencoderTrainer:

    def __init__(self, cfg):

        self.cfg = cfg
        self.model = SparseAutoencoder(cfg.input_dim, cfg.hidden_dim, cfg.l1_lambda, cfg.tied_weights).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.best_val_mse = float('inf')
        self.output_dir, self.log_path = self._build_log_dir()
        self._log_config()

    def _build_log_dir(self):

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f'model/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'training_log.txt')
        return output_dir, log_path

    def _log_config(self):
        self.log("Training Configuration:")
        for key, value in vars(self.cfg).items():
            self.log(f"  {key}: {value}")


    def log(self, message):
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')

    def load_data(self, data_dir='./data/dataset.json'):
        with open(data_dir, 'r') as file:
            dataset_raw = json.load(file)

        dataset = [v2_item[1] for v1 in dataset_raw.values() for v2 in v1.values() for v2_item in v2]
        data_tensor = torch.tensor(dataset, dtype=torch.float32)

        # train, test = train_test_split(data_tensor, test_size=0.3, random_state=42)
        # val, test = train_test_split(test, test_size=0.67, random_state=42)

        train, val = train_test_split(val, test_size=0.1, random_state=42)

        self.train_loader = DataLoader(TensorDataset(train), batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(TensorDataset(val), batch_size=self.cfg.batch_size, shuffle=False, drop_last=True)

        self.log(f"Train shape: {train.shape}")
        self.log(f"Val shape: {val.shape}")
        self.log(f"Test shape: {test.shape}")

    def train(self):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss, epoch_mse, epoch_l1, epoch_l0 = 0, 0, 0, 0

            for batch in self.train_loader:
                inputs = batch[0].cuda()
                targets = inputs.clone()

                self.optimizer.zero_grad()
                outputs, z = self.model(inputs)
                loss, mse_loss, l1_loss, l0_loss = self.model.loss(outputs, targets, z)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse_loss.item()
                epoch_l1 += l1_loss.item()
                epoch_l0 += l0_loss.item()

            self.log(f"[Train] Epoch {epoch+1}/{self.cfg.epochs}, "
                     f"MSE: {epoch_mse/len(self.train_loader):.6f}, "
                     f"L1: {epoch_l1/len(self.train_loader):.6f}, "
                     f"L0: {epoch_l0/len(self.train_loader):.6f}, "
                     f"Total: {epoch_loss/len(self.train_loader):.6f}")

            self.evaluate(epoch)

        self.log("Training complete. Final model saved.")

    def evaluate(self, epoch):
        self.model.eval()
        total_mse, total_l1, total_l0 = 0, 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch[0].cuda()
                targets = inputs.clone()
                outputs, z = self.model(inputs)
                _, mse_loss, l1_loss, l0_loss = self.model.loss(outputs, targets, z)
                total_mse += mse_loss.item()
                total_l1 += l1_loss.item()
                total_l0 += l0_loss.item()

        avg_mse = total_mse / len(self.val_loader)
        avg_l1 = total_l1 / len(self.val_loader)
        avg_l0 = total_l0 / len(self.val_loader)

        self.log(f"[ Val ] Epoch {epoch+1}/{self.cfg.epochs}, "
                 f"MSE: {avg_mse:.6f}, "
                 f"L1: {avg_l1:.6f}, "
                 f"L0: {avg_l0:.6f}, ")

        if avg_mse < self.best_val_mse:
            self.best_val_mse = avg_mse
            torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.output_dir, 'best_model.pth'))
            self.log(f"Epoch {epoch+1}: Saved new best model with MSE {self.best_val_mse:.6f}")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--input_dim', type=int, default=896)
    parser.add_argument('--hidden_dim', type=int, default=896 * 20)
    parser.add_argument('--l1_lambda', type=float, default=4e-8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tied_weights', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':

    cfg = get_config()
    trainer = SparseAutoencoderTrainer(cfg)
    trainer.load_data()
    trainer.train()
