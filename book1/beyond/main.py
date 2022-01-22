import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.nn import Module

from book1.beyond.datasets.dataLoader import get_mnist_handwritten
from book1.beyond.graphs.models.LeNet_torch import LeNet1Torch, LeNet4Torch, LeNet5Torch


class BaseTrainer:
    def __init__(self,
                 which: str,
                 RANDOM_SEED: int = 42,
                 LEARNING_RATE: float = 0.01,
                 BATCH_SIZE: int = 100,
                 N_EPOCHS: int = 20,
                 IMG_SIZE: int = 32,
                 N_CLASSES: int = 10,
                 ):
        self.models = {
            'LeNet1_torch': LeNet1Torch(),
            'LeNet4_torch': LeNet4Torch(),
            'LeNet5_torch': LeNet5Torch(),
        }
        self.model = self.models[which]
        self.data = get_mnist_handwritten(as_dataloader=True)

        self.mini_batch_size = self.batch_size = BATCH_SIZE
        self.epochs = N_EPOCHS

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.learning_rate = LEARNING_RATE

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def get_net(self):
        print("# Net")
        print(self.model)

        print("# Params")
        for param in self.model.parameters():
            print(param.size())

    def train_step(self, x_batch, y_batch, batch_idx, total):
        # Compute prediction and loss
        pred = self.model(x_batch)
        loss = self.loss_fn(pred, y_batch)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(x_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")

    def train_loop(self):
        total = len(self.data.train.dataset)

        for batch, (X, y) in enumerate(self.data.train):
            self.train_step(X, y, batch, total)

    def train(self):
        for _ in range(self.epochs):
            self.train_loop()

        print('Finished Training')

    def eval(self, model: str):
        self.models[model].eval()

    def infer(self, model: str):
        self.models[model].infer()


if __name__ == '__main__':
    # ======== Device Setting ========= #
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =========== Parameters ========== #
    RANDOM_SEED = 42
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    N_EPOCHS = 15

    IMG_SIZE = 32
    N_CLASSES = 10

    # ========== Run Setting ========== #
    target_mode = "train"
    target_model = "LeNet1_torch"

    # ================================= #

    base_trainer = BaseTrainer(
        which=target_model,
        RANDOM_SEED=42,
        LEARNING_RATE=0.001,
        BATCH_SIZE=32,
        N_EPOCHS=15,
        IMG_SIZE=32,
        N_CLASSES=10,
    )
    base_trainer.get_net()
    base_trainer.train()
