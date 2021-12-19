import torch
from torch import nn
import torch.optim as optim

from book1.beyond.datasets.dataLoader import get_mnist_handwritten
from book1.beyond.graphs.models.LeNet_torch import LeNet1Torch, LeNet4Torch, LeNet5Torch


class BaseTrainer:
    def __init__(self):
        self.models = {
            'LeNet1_torch': LeNet1Torch(),
            'LeNet4_torch': LeNet4Torch(),
            'LeNet5_torch': LeNet5Torch(),
        }
        self.data = get_mnist_handwritten(as_data=True)

    def get_net(self, model):
        print("# Net")
        print(self.models[model])

        print("# Params")
        for param in self.models[model].parameters():
            print(param.size())

    def train(self, model: str):

        cur_model = self.models[model].train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cur_model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = cur_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

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
    target_mode = "get_net"
    target_model = "LeNet1_torch"

    # ================================= #

    base_trainer = BaseTrainer()
    getattr(base_trainer, target_mode)(target_model)
