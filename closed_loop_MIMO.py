import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import scipy.io as io
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from torch.optim import SGD, Adam
from torch.utils import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_snr = 15
M = 16
NN_T = 64
NN_R = 512

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.fc1 = nn.Linear(M, NN_T)
        self.fc2 = nn.Linear(NN_T, NN_T)
        self.fc3 = nn.Linear(NN_T, 8)

    def forward(self, in_message):
        x = F.relu(self.fc1(in_message))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        transmitted = x / torch.sqrt(2*torch.mean(x**2))

        return transmitted

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(4+4, NN_R)
        self.fc2 = nn.Linear(NN_R, NN_R)
        self.fc3 = nn.Linear(NN_R, NN_R)
        self.fc4 = nn.Linear(NN_R, M)

    def forward(self, in_message, in_channel):
        nn_input = torch.cat([in_message, in_channel],-1)
        x = F.relu(self.fc1(nn_input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        decoded = F.log_softmax(self.fc4(x), dim=-1)

        return decoded

encoder = Encoder().to(device)
decoder = Decoder().to(device)

criterion = nn.NLLLoss()
opt = Adam(list(encoder.parameters())+list(decoder.parameters()), lr=0.001)

loss = np.array([])
batch = 200000
# batch = 2000

csi_real_set = torch.randn((M * batch, 2, 1)) / np.sqrt(2)
csi_imag_set = torch.randn((M * batch, 2, 1)) / np.sqrt(2)

data_set = data.TensorDataset(csi_real_set, csi_imag_set)
training_generator = data.DataLoader(data_set, batch_size=2048, shuffle=True)

for epoches in np.arange(20):
    print('epoch =', epoches)
    for ch_real, ch_imag in training_generator:
        csi_real = ch_real.to(device)
        csi_imag = ch_imag.to(device)

        messages = np.random.randint(0, M, csi_real.shape[0])

        train_data = torch.from_numpy(to_categorical(messages)).to(device)
        train_label = torch.from_numpy(messages).long().to(device)

        transmitted = encoder(train_data)
        tx_real = transmitted[:, np.arange(0, 4)].view(-1, 2, 2)
        tx_imag = transmitted[:, np.arange(4, 8)].view(-1, 2, 2)

        rx_real = torch.bmm(tx_real, csi_real) - torch.bmm(tx_imag, csi_imag)
        rx_imag = torch.bmm(tx_real, csi_imag) + torch.bmm(tx_imag, csi_real)

        rx = torch.cat([rx_real, rx_imag], axis=-2).view(-1, 4)

        sigma = np.sqrt(0.5 / (np.power(10, train_snr / 10)))
        noise = (sigma * torch.randn(rx.shape)).to(device)
        rx = rx + noise

        csi = torch.cat([csi_real, csi_imag], axis=-2).view(-1, 4)

        y_pred = decoder(rx, csi)
        cross_entropy = criterion(y_pred, train_label)

        opt.zero_grad()

        cross_entropy.backward()
        opt.step()

    l = cross_entropy.to('cpu').detach().numpy()
    loss = np.append(loss, l)

