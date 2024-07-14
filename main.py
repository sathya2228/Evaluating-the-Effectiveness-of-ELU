import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets.mnist import load_data

sns.set_theme()

# load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 127.5 - 1
testX = np.float32(testX) / 127.5 - 1

# Convert numpy arrays to torch tensors
trainX = torch.tensor(trainX).reshape(-1, 1, 28, 28)
trainy = torch.tensor(trainy).long()
testX = torch.tensor(testX).reshape(-1, 1, 28, 28)
testy = torch.tensor(testy).long()

# Prepare DataLoader
dataset = DataLoader(TensorDataset(trainX, trainy), batch_size=64, shuffle=True)
testing_dataset = DataLoader(TensorDataset(testX, testy), batch_size=64, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ELU(nn.Module):
    def _init_(self, alpha=1.):
        super(ELU, self)._init_()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

class CNN(nn.Module):
    def _init_(self, activation_function):
        super(CNN, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = activation_function

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

def train(model, optimizer, loss_fct=torch.nn.NLLLoss(), nb_epochs=25):
    training_loss = []
    validation_loss = []
    for epoch in tqdm(range(nb_epochs)):
        model.train()
        batch_loss = []
        for batch_idx, (x, y) in enumerate(dataset):
            x = x.to(device)
            y = y.to(device)

            log_prob = model(x)
            loss = loss_fct(log_prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

            # Log training details
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{nb_epochs}], Batch [{batch_idx+1}/{len(dataset)}], Loss: {loss.item():.4f}')

        training_loss.append(np.mean(batch_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            x = testX.to(device)
            y = testy.to(device)
            log_prob = model(x)
            t_loss = loss_fct(log_prob, y)
            validation_loss.append(t_loss.item())

        print(f'Epoch [{epoch+1}/{nb_epochs}] completed. Training Loss: {training_loss[-1]:.4f}, Validation Loss: {validation_loss[-1]:.4f}')

    return training_loss, validation_loss

if _name_ == "_main_":
    # ReLU Model
    model = CNN(nn.ReLU()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    training_loss, validation_loss = train(model, optimizer, nb_epochs=25)
    plt.plot(training_loss, label='ReLU Training')
    plt.plot(validation_loss, label='ReLU Validation', linestyle='--')

    # ELU Model
    model = CNN(ELU()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    training_loss, validation_loss = train(model, optimizer, nb_epochs=25)
    plt.plot(training_loss, label='ELU Training')
    plt.plot(validation_loss, label='ELU Validation', linestyle='--')

    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Cross Entropy Loss', fontsize=14)
    plt.savefig('cnn_elu_vs_relu.png')
    plt.show()