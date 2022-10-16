import time
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import gc

import torch
from torch import device, nn, reshape, functional, softmax
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.optim import lr_scheduler

from conduct_model import Model


torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

n_img = 200
n_output = 200


def getNoise(x_V, SNR):
    tar_V = np.zeros(x_V.shape)
    Noise = np.random.randn(x_V.shape[0], x_V.shape[1])
    Noise = Noise - np.mean(Noise)
    sig_pow = 1/len(x_V)*np.sum(np.multiply(x_V, x_V))
    Noise_vari = sig_pow/(10**(SNR/10))
    Noise = np.sqrt(Noise_vari)/np.std(Noise)*Noise
    tar_V = x_V + Noise
    return tar_V


path = './Data/random_200/'
x_train = np.load(path + 'x_train.npy')
mask_train = np.load(path+'mask_train.npy')[:, :, 0, :]
amount_train = mask_train[:, :, 0].sum(axis=1)
x_train = np.concatenate((x_train, getNoise(x_train[:6000], 40), getNoise(x_train[6000:], 50)))
cdt_train = np.load(path + 'cdt_train.npy')
cdt_train[np.where(cdt_train == 0.1)] = 1
cdt_train[np.where(cdt_train == 3.9)] = 2
cdt_train = np.concatenate((cdt_train, cdt_train[:6000], cdt_train[6000:]))
num1 = np.array(np.where(amount_train == 1))

x_val = np.load(path + 'x_val.npy')
x_val = np.concatenate((x_val, getNoise(x_val[:2113], 40), getNoise(x_val[2113:], 50)))
cdt_val = np.load(path + 'cdt_val.npy')
cdt_val[np.where(cdt_val == 0.1)] = 1
cdt_val[np.where(cdt_val == 3.9)] = 2
cdt_val = np.concatenate((cdt_val, cdt_val[:2113], cdt_val[2113:]))
mask_val = np.load(path+'mask_val.npy')[:, :, 0, :]
amount_val = mask_val[:, :, 0].sum(axis=1)
num1 = np.array(np.where(amount_val == 1))

x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')
cdt_test = np.load(path + 'cdt_test.npy')
cdt_test[np.where(cdt_test == 0.1)] = 1
cdt_test[np.where(cdt_test == 3.9)] = 2
mask_test = np.load(path+'mask_test.npy')[:, :, 0, :]
amount_test = mask_test[:, :, 0].sum(axis=1)
num1 = np.array(np.where(amount_test == 1))

# Create the dataset
class SimDataset(Dataset):
    def __init__(self, voltages, conductivity, transform=None):
        self.voltages, self.conductivity = voltages, conductivity
        self.transform = transform

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, idx):
        voltage = self.voltages[idx].reshape(1, 208)  # bz, 208, 1
        conduct = self.conductivity[idx]

        return [voltage, conduct]


trans = transforms.Compose([
    transforms.ToTensor(),
])


batch_size = 32

datasets = {
    "train": SimDataset(x_train, cdt_train, transform=trans),
    "valid": SimDataset(x_val, cdt_val, transform=trans),
    "test": SimDataset(x_test, cdt_test, transform=trans)
}


dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
    "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=True),
    "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
}

transform2 = transforms.Compose([
    transforms.ToTensor(),
])

path2 = "./Data/"

# create and normalize the pos and initial point
sensor_pos = np.load(path2 + 'sensor_208.npy').reshape(208, 1, 6)
sensor_pos = torch.from_numpy(sensor_pos).float().to(device)  # [208, 1, 6]

n_point = 3
n_embed = 128
n_hid = 256
n_head = 4
n_elayers = 6
dropout = 0
lr = 1e-4
n_mlayers = None
n_dlayers = 6
m = 24
epoch = 1000


model = Model(n_point=n_point, n_embed=n_embed, n_hid=n_hid, n_head=n_head, n_elayers=n_elayers, n_mlayers=n_mlayers,
              n_dlayers= n_dlayers, dropout=dropout
              )
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = 25, gamma = 0.95)

model = model.to(device)


loss_fn = nn.CrossEntropyLoss()


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


# calculate the accuracy
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


train_loss_plot = []
val_loss_plot = []
train_acc_plot = []
val_acc_plot = []

# train the model
for i in range(epoch):
    since = time.time()
    train_lossall = 0
    epoch_samples = 0
    model = model.train()
    with tqdm(dataloaders['train'], unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {i}/{epoch}")
            vols, cdt = data
            vols = vols.permute(1, 0, 2).float().to(device) # 1, bz, 208
            cdt = cdt.long().to(device)
            optimizer.zero_grad()
            output = model.forward(sensor_pos, vols, batch_size=vols.size(1))
            a = output.reshape(-1, 3)
            b = cdt.reshape(-1)
            loss = loss_fn(a, b)
            acc = accuracy(a, b)/len(b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_lossall += loss.detach().item() * vols.size(1)
            epoch_samples += vols.size(1)
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            tepoch.set_postfix(TrainLoss_all=train_lossall / epoch_samples, lr=lr, Train_acc=acc)
    scheduler.step()
    train_loss_plot.append(train_lossall / epoch_samples)
    train_acc_plot.append(acc)

    vali_loss = 0
    epoch_samples = 0
    best_val_loss = float('inf')
    model.eval()
    with torch.no_grad():
        with tqdm(dataloaders['valid'], unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {i}/{epoch}")
                vols, cdt = data
                vols = vols.permute(1, 0, 2).float().to(device)
                cdt = cdt.long().to(device)
                output = model.forward(sensor_pos, vols, batch_size=vols.size(1))
                a = output.reshape(-1, 3)
                b = cdt.reshape(-1)
                loss = loss_fn(a, b)
                acc = accuracy(a, b) / len(b)
                vali_loss += loss.item() * vols.size(1)
                epoch_samples += vols.size(1)

                tepoch.set_postfix(ValidLoss=vali_loss / epoch_samples, Valid_acc=acc)

        val_loss_plot.append(vali_loss / epoch_samples)
        val_acc_plot.append(acc)

        # if train_lossall < best_val_loss:
        #     best_val_loss = train_lossall
        #     torch.save(model.state_dict(), './saved_Models/0308_cdt_model.pt')
        #     sleep(0.1)
        #
        # np.save('./loss_curve/0308_train_loss_cdt.npy', train_loss_plot)
        # np.save('./loss_curve/0308_val_loss_cdt.npy', val_loss_plot)
        # np.save('./loss_curve/0308_train_acc_cdt.npy', train_acc_plot)
        # np.save('./loss_curve/0308_val_acc_cdt.npy', val_acc_plot)

print('training end')