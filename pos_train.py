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

from pos_model import Model

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
mask_train = np.concatenate((mask_train, mask_train[:6000], mask_train[6000:]))
x_train = np.concatenate((x_train, getNoise(x_train[:6000], 40), getNoise(x_train[6000:], 50)))
pos_train = np.load(path + 'pos_train.npy')
pos_train[:, :, 2] = pos_train[:, :, 2]/15
pos_train[:, :, :2] = pos_train[:, :, :2]/(11.48*2)
pos_train = np.concatenate((pos_train, pos_train[:6000], pos_train[6000:]))


x_val = np.load(path + 'x_val.npy')
mask_val = np.load(path + 'mask_val.npy')[:, :, 0, :]
mask_val = np.concatenate((mask_val, mask_val[:2113], mask_val[2113:]))
x_val = np.concatenate((x_val, getNoise(x_val[:2113], 40), getNoise(x_val[2113:], 50)))
pos_val = np.load(path + 'pos_val.npy')
pos_val[:, :, 2] = pos_val[:, :, 2]/15
pos_val[:, :, :2] = pos_val[:, :, :2]/(11.48*2)
pos_val = np.concatenate((pos_val, pos_val[:2113], pos_val[2113:]))


x_test = np.load(path + 'x_test.npy')
mask_test = np.load(path + 'mask_test.npy')[:, :, 0, :]
y_test = np.load(path + 'y_test.npy')
pos_test = np.load(path + 'pos_test.npy')
pos_test[:, :, 2] = pos_test[:, :, 2]/15
pos_test[:, :, :2] = pos_test[:, :, :2]/(11.48*2)


# Create the dataset
class SimDataset(Dataset):
    def __init__(self, voltages, pos, mask_pos,  transform=None):
        self.voltages, self.pos, self.mask_pos = voltages, pos, mask_pos
        self.transform = transform

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, idx):
        voltage = self.voltages[idx].reshape(1, 208)  # bz, 208, 1
        pos = self.pos[idx]
        mask_pos = self.mask_pos[idx]

        return [voltage, pos, mask_pos]


trans = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 32

datasets = {
    "train": SimDataset(x_train, pos_train, mask_train, transform=trans),
    "valid": SimDataset(x_val, pos_val, mask_val, transform=trans),
    "test": SimDataset(x_test, pos_test, mask_test, transform=trans)
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

loss_fn = nn.MSELoss()


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


# calculate the accuracy
def accuracy(y_hat, y):  #@save
    """???????????????????????????"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


train_loss_plot = []
val_loss_plot = []

# train the model
for i in range(epoch):
    since = time.time()
    train_lossall = 0
    epoch_samples = 0
    model = model.train()
    with tqdm(dataloaders['train'], unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {i}/{epoch}")
            vols, pos, mask_pos = data
            vols = vols.permute(1, 0, 2).float().to(device) # 1, bz, 208
            mask_pos = mask_pos.float().to(device)
            pos = pos.float().to(device)
            optimizer.zero_grad()
            output = model.forward(sensor_pos, vols, batch_size=vols.size(1))
            # a = output.reshape(-1, 3)
            # b = cdt.reshape(-1)
            loss = loss_fn(output * mask_pos, pos * mask_pos)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_lossall += loss.detach().item() * vols.size(1)
            epoch_samples += vols.size(1)
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            tepoch.set_postfix(TrainLoss_all=train_lossall / epoch_samples, lr=lr)
    scheduler.step()
    train_loss_plot.append(train_lossall / epoch_samples)

    vali_loss = 0
    epoch_samples = 0
    best_val_loss = float('inf')
    model.eval()
    with torch.no_grad():
        with tqdm(dataloaders['valid'], unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {i}/{epoch}")
                vols, pos, mask_pos = data
                vols = vols.permute(1, 0, 2).float().to(device)
                mask_pos = mask_pos.float().to(device)
                pos = pos.float().to(device)
                output = model.forward(sensor_pos, vols, batch_size=vols.size(1))
                loss = loss_fn(output * mask_pos, pos * mask_pos)
                vali_loss += loss.item() * vols.size(1)
                epoch_samples += vols.size(1)
                tepoch.set_postfix(ValidLoss=vali_loss / epoch_samples)

        val_loss_plot.append(vali_loss / epoch_samples)

        # if vali_loss < best_val_loss:
        #     best_val_loss = vali_loss
        #     torch.save(model.state_dict(), './saved_Models/0315_pos_model.pt')
        #     sleep(0.1)
        #
        # np.save('./loss_curve/0315_train_loss_pos.npy', train_loss_plot)
        # np.save('./loss_curve/0315_val_loss_pos.npy', val_loss_plot)

print('training end')