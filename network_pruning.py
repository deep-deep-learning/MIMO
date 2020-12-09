import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from keras.utils import to_categorical

encoder_Si = {'fc1': np.load('First Try/encoder_Si_fc1.npy'), 'fc2': np.load('First Try/encoder_Si_fc2.npy'), 'fc3': np.load(
    'First Try/encoder_Si_fc3.npy')}

insignificant_idx = {'fc1': np.where(encoder_Si['fc1'] < 1e-4)[0], 'fc2': np.where(encoder_Si['fc2'] < 1e-4)[0], 'fc3': np.where(encoder_Si['fc3'] < 1e-4)[0]}
significant_idx = {'fc1': np.where(encoder_Si['fc1'] >= 1e-4)[0], 'fc2': np.where(encoder_Si['fc2'] >= 1e-4)[0], 'fc3': np.where(encoder_Si['fc3'] >= 1e-4)[0]}

train_snr = 15
M = 16
NN_T = 64
NN_R = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

encoder.load_state_dict(torch.load('encoder.pt'))

# Initialize a reduced model
class reduced_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(reduced_Encoder,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 8)

    def forward(self, in_message):
        x = F.relu(self.fc1(in_message))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        transmitted = x / torch.sqrt(2*torch.mean(x**2))

        return transmitted

input_size = M - len(insignificant_idx['fc1'])
hidden_size_1 = NN_T - len(insignificant_idx['fc2'])
hidden_size_2 = NN_T - len(insignificant_idx['fc3'])
reduced_encoder = reduced_Encoder(input_size, hidden_size_1, hidden_size_2)

x = encoder.fc1.weight.data
x = torch.index_select(x, 1, torch.from_numpy(significant_idx['fc1']))
x = torch.index_select(x, 0, torch.from_numpy(significant_idx['fc2']))
reduced_encoder.fc1.weight.data = x

x = encoder.fc1.bias.data
x = torch.index_select(x, 0, torch.from_numpy(significant_idx['fc2']))
reduced_encoder.fc1.bias.data = x

x = encoder.fc2.weight.data
x = torch.index_select(x, 1, torch.from_numpy(significant_idx['fc2']))
x = torch.index_select(x, 0, torch.from_numpy(significant_idx['fc3']))
reduced_encoder.fc2.weight.data = x

x = encoder.fc2.bias.data
x = torch.index_select(x, 0, torch.from_numpy(significant_idx['fc3']))
reduced_encoder.fc2.bias.data = x

x = encoder.fc3.weight.data
x = torch.index_select(x, 1, torch.from_numpy(significant_idx['fc3']))
reduced_encoder.fc3.weight.data = x

reduced_encoder.fc3.bias.data = encoder.fc3.bias.data

SER = np.array([])
SER_ = np.array([])
with torch.no_grad():
    for test_snr in np.arange(0,26,2):
        ser_temp = np.array([])
        ser_temp_ = np.array([])
        for temp in np.arange(100):
            batch = 40
            messages = np.arange(M)
            messages = np.tile(messages, batch)
            test_labels = to_categorical(messages)

            csi_real = (torch.randn((M*batch, 2,1))/np.sqrt(2)).to(device)
            csi_imag = (torch.randn((M*batch, 2,1))/np.sqrt(2)).to(device)

            test_data = torch.from_numpy(test_labels).to(device)
            test_label = torch.from_numpy(messages).to(device)

            transmitted = encoder(test_data)
            tx_real = transmitted[:, np.arange(0,4)].view(-1,2,2)
            tx_imag = transmitted[:, np.arange(4,8)].view(-1,2,2)

            rx_real = torch.bmm(tx_real, csi_real) - torch.bmm(tx_imag, csi_imag)
            rx_imag = torch.bmm(tx_real, csi_imag) + torch.bmm(tx_imag, csi_real)

            rx = torch.cat([rx_real, rx_imag], axis=-2).view(-1,4)

            transmitted_ = reduced_encoder(test_data)
            tx_real = transmitted_[:, np.arange(0, 4)].view(-1, 2, 2)
            tx_imag = transmitted_[:, np.arange(4, 8)].view(-1, 2, 2)

            rx_real = torch.bmm(tx_real, csi_real) - torch.bmm(tx_imag, csi_imag)
            rx_imag = torch.bmm(tx_real, csi_imag) + torch.bmm(tx_imag, csi_real)

            rx_ = torch.cat([rx_real, rx_imag], axis=-2).view(-1, 4)

            sigma = np.sqrt(0.5/(np.power(10, test_snr/10)))
            noise = (sigma * torch.randn(rx.shape)).to(device)
            rx = rx + noise
            rx_ = rx_ + noise

            csi = torch.cat([csi_real, csi_imag], axis=-2).view(-1,4)

            y_pred = decoder(rx, csi)
            classification = torch.argmax(y_pred, axis=-1).to('cpu').detach().numpy()
            correct = np.equal(classification, messages)
            ser = 1 - np.mean(correct)
            ser_temp = np.append(ser_temp, ser)

            y_pred_ = decoder(rx_, csi)
            classification = torch.argmax(y_pred_, axis=-1).to('cpu').detach().numpy()
            correct = np.equal(classification, messages)
            ser = 1 - np.mean(correct)
            ser_temp_ = np.append(ser_temp_, ser)

        ser_ave = np.mean(ser_temp)
        SER = np.append(SER, ser_ave)
        ser_ave = np.mean(ser_temp_)
        SER_ = np.append(SER_, ser_ave)

np.save('First Try/SER_full_model.npy', SER)
np.save('First Try/SER_reduced_model.npy', SER_)

