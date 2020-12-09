import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from keras.utils import to_categorical
from torch.optim import SGD, Adam
from torch.utils import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

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

encoder_inputs = {'fc1':[], 'fc2':[], 'fc3':[]}
decoder_inputs = {'fc1':[], 'fc2':[], 'fc3':[], 'fc4':[]}
def collect_encoder_inputs_fc1(self, input, output):
    encoder_inputs['fc1'].append(input[0].detach())
def collect_encoder_inputs_fc2(self, input, output):
    encoder_inputs['fc2'].append(input[0].detach())
def collect_encoder_inputs_fc3(self, input, output):
    encoder_inputs['fc3'].append(input[0].detach())

def collect_decoder_inputs_fc1(self, input, output):
    decoder_inputs['fc1'].append(input[0].detach())
def collect_decoder_inputs_fc2(self, input, output):
    decoder_inputs['fc2'].append(input[0].detach())
def collect_decoder_inputs_fc3(self, input, output):
    decoder_inputs['fc3'].append(input[0].detach())
def collect_decoder_inputs_fc4(self, input, output):
    decoder_inputs['fc4'].append(input[0].detach())

encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder.fc1.register_forward_hook(collect_encoder_inputs_fc1)
encoder.fc2.register_forward_hook(collect_encoder_inputs_fc2)
encoder.fc3.register_forward_hook(collect_encoder_inputs_fc3)

decoder.fc1.register_forward_hook(collect_decoder_inputs_fc1)
decoder.fc2.register_forward_hook(collect_decoder_inputs_fc2)
decoder.fc3.register_forward_hook(collect_decoder_inputs_fc3)
decoder.fc4.register_forward_hook(collect_decoder_inputs_fc4)

criterion = nn.NLLLoss()
opt = Adam(list(encoder.parameters())+list(decoder.parameters()), lr=0.001)

loss = np.array([])
batch = 2048
# batch = 2000

csi_real_set = torch.randn((M * batch, 2, 1)) / np.sqrt(2)
csi_imag_set = torch.randn((M * batch, 2, 1)) / np.sqrt(2)

data_set = data.TensorDataset(csi_real_set, csi_imag_set)
batch_size = 2048
training_generator = data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

for epoches in np.arange(2):
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

def compute_sobol_indices(function, inputs, device):
    with torch.no_grad():
        Si = np.zeros((inputs[0].shape[1]))
        N = int(inputs[0].shape[0]/2)
        for i in range(len(inputs)):
            A = inputs[i][:N, :].to(device)
            B = inputs[i][N:, :]
            for d in range(inputs[0].shape[1]):
                A_ = A.detach().clone().to(device)
                A_[:, d] = B[:, d]
                outputs_ = function(A_)
                outputs = function(A)
                Si[d] = 1 / (2 * N) * torch.sum((torch.mean(outputs, axis=1) - torch.mean(outputs_, axis=1)) ** 2)

    return Si

num_samples = 400
encoder_Si = {}

fc1_test_encoder = Encoder().to(device)
fc1_test_encoder.fc1.weight.data = encoder.fc1.weight.data
fc1_test_encoder.fc1.bias.data = encoder.fc1.bias.data
fc1_test_encoder.fc2.weight.data = encoder.fc2.weight.data
fc1_test_encoder.fc2.bias.data = encoder.fc2.bias.data
fc1_test_encoder.fc3.weight.data = encoder.fc3.weight.data
fc1_test_encoder.fc3.bias.data = encoder.fc3.bias.data

inputs = encoder_inputs['fc1'][:num_samples]
encoder_Si['fc1'] = compute_sobol_indices(fc1_test_encoder, inputs, device)

class Encoder_fc2(nn.Module):
    def __init__(self):
        super(Encoder_fc2,self).__init__()
        self.fc2 = nn.Linear(NN_T, NN_T)
        self.fc3 = nn.Linear(NN_T, 8)

    def forward(self, input):
        x = F.relu(self.fc2(input))
        x = self.fc3(x)
        transmitted = x / torch.sqrt(2*torch.mean(x**2))

        return transmitted

fc2_test_encoder = Encoder_fc2().to(device)
fc2_test_encoder.fc2.weight.data = encoder.fc2.weight.data
fc2_test_encoder.fc2.bias.data = encoder.fc2.bias.data
fc2_test_encoder.fc3.weight.data = encoder.fc3.weight.data
fc2_test_encoder.fc3.bias.data = encoder.fc3.bias.data

inputs = encoder_inputs['fc2'][:num_samples]
encoder_Si['fc2'] = compute_sobol_indices(fc2_test_encoder, inputs, device)

class Encoder_fc3(nn.Module):
    def __init__(self):
        super(Encoder_fc3,self).__init__()
        self.fc3 = nn.Linear(NN_T, 8)

    def forward(self, input):
        x = self.fc3(input)
        transmitted = x / torch.sqrt(2*torch.mean(x**2))

        return transmitted

fc3_test_encoder = Encoder_fc3().to(device)
fc3_test_encoder.fc3.weight.data = encoder.fc3.weight.data
fc3_test_encoder.fc3.bias.data = encoder.fc3.bias.data

inputs = encoder_inputs['fc3'][:num_samples]
encoder_Si['fc3'] = compute_sobol_indices(fc3_test_encoder, inputs, device)

class Decoder_fc1(nn.Module):
    def __init__(self):
        super(Decoder_fc1, self).__init__()
        self.fc1 = nn.Linear(4+4, NN_R)
        self.fc2 = nn.Linear(NN_R, NN_R)
        self.fc3 = nn.Linear(NN_R, NN_R)
        self.fc4 = nn.Linear(NN_R, M)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        decoded = F.log_softmax(self.fc4(x), dim=-1)

        return decoded

fc1_test_decoder = Decoder_fc1().to(device)
fc1_test_decoder.fc1.weight.data = decoder.fc1.weight.data
fc1_test_decoder.fc1.bias.data = decoder.fc1.bias.data
fc1_test_decoder.fc2.weight.data = decoder.fc2.weight.data
fc1_test_decoder.fc2.bias.data = decoder.fc2.bias.data
fc1_test_decoder.fc3.weight.data = decoder.fc3.weight.data
fc1_test_decoder.fc3.bias.data = decoder.fc3.bias.data
fc1_test_decoder.fc4.weight.data = decoder.fc4.weight.data
fc1_test_decoder.fc4.bias.data = decoder.fc4.bias.data

decoder_Si = {}

inputs = decoder_inputs['fc1'][:num_samples]
decoder_Si['fc1'] = compute_sobol_indices(fc1_test_decoder, inputs, device)

print("here")

class Decoder_fc2(nn.Module):
    def __init__(self):
        super(Decoder_fc2, self).__init__()
        self.fc2 = nn.Linear(NN_R, NN_R)
        self.fc3 = nn.Linear(NN_R, NN_R)
        self.fc4 = nn.Linear(NN_R, M)

    def forward(self, input):
        x = F.relu(self.fc2(input))
        x = F.relu(self.fc3(x))
        decoded = F.log_softmax(self.fc4(x), dim=-1)

        return decoded

fc2_test_decoder = Decoder_fc2().to(device)
fc2_test_decoder.fc2.weight.data = decoder.fc2.weight.data
fc2_test_decoder.fc2.bias.data = decoder.fc2.bias.data
fc2_test_decoder.fc3.weight.data = decoder.fc3.weight.data
fc2_test_decoder.fc3.bias.data = decoder.fc3.bias.data
fc2_test_decoder.fc4.weight.data = decoder.fc4.weight.data
fc2_test_decoder.fc4.bias.data = decoder.fc4.bias.data

inputs = decoder_inputs['fc2'][:num_samples]
decoder_Si['fc2'] = compute_sobol_indices(fc2_test_decoder, inputs, device)

class Decoder_fc3(nn.Module):
    def __init__(self):
        super(Decoder_fc3, self).__init__()
        self.fc3 = nn.Linear(NN_R, NN_R)
        self.fc4 = nn.Linear(NN_R, M)

    def forward(self, input):
        x = F.relu(self.fc3(input))
        decoded = F.log_softmax(self.fc4(x), dim=-1)

        return decoded

fc3_test_decoder = Decoder_fc3().to(device)
fc3_test_decoder.fc3.weight.data = decoder.fc3.weight.data
fc3_test_decoder.fc3.bias.data = decoder.fc3.bias.data
fc3_test_decoder.fc4.weight.data = decoder.fc4.weight.data
fc3_test_decoder.fc4.bias.data = decoder.fc4.bias.data

inputs = decoder_inputs['fc3'][:num_samples]
decoder_Si['fc3'] = compute_sobol_indices(fc3_test_decoder, inputs, device)

class Decoder_fc4(nn.Module):
    def __init__(self):
        super(Decoder_fc4, self).__init__()
        self.fc4 = nn.Linear(NN_R, M)

    def forward(self, input):
        decoded = F.log_softmax(self.fc4(input), dim=-1)

        return decoded

fc4_test_decoder = Decoder_fc4().to(device)
fc4_test_decoder.fc4.weight.data = decoder.fc4.weight.data
fc4_test_decoder.fc4.bias.data = decoder.fc4.bias.data

inputs = decoder_inputs['fc4'][:num_samples]
decoder_Si['fc4'] = compute_sobol_indices(fc4_test_decoder, inputs, device)

torch.save(encoder.state_dict(), 'encoder.pt')
torch.save(decoder.state_dict(), 'decoder.pt')

np.save('encoder_Si_fc1.npy', encoder_Si['fc1'])
np.save('encoder_Si_fc2.npy', encoder_Si['fc2'])
np.save('encoder_Si_fc3.npy', encoder_Si['fc3'])

np.save('decoder_Si_fc1.npy', decoder_Si['fc1'])
np.save('decoder_Si_fc2.npy', decoder_Si['fc2'])
np.save('decoder_Si_fc3.npy', decoder_Si['fc3'])
np.save('decoder_Si_fc4.npy', decoder_Si['fc4'])