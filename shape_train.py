# The code file is part of my undergraduate senior project and written by myself.
import time
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import gc
from random import randint

import torch
from torch import device, nn, reshape, functional, softmax
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, Normalizer
from torch.optim import lr_scheduler

from shape_model import Model
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from function import marsagliasample, blocksample, marsagliasample2, blocksample2

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
y_train = np.load(path + 'y_train.npy')
mask_train = np.load(path + 'mask_train.npy')[:, :, 0, :]
mask_train_img = np.repeat(np.expand_dims(mask_train, 2), n_img, 2)
mask_train_out = np.repeat(np.expand_dims(mask_train, 2), n_output, 2)
init_train = np.load(path + 'init_train.npy')
x_train = np.concatenate((x_train, getNoise(x_train[:6000], 40), getNoise(x_train[6000:], 50)))
y_train = np.concatenate((y_train, y_train[:6000], y_train[6000:]))
mask_train = np.concatenate((mask_train, mask_train[:6000], mask_train[6000:]))
mask_train_img = np.concatenate((mask_train_img, mask_train_img[:6000], mask_train_img[6000:]))
mask_train_out = np.concatenate((mask_train_out, mask_train_out[:6000], mask_train_out[6000:]))
init_train = np.concatenate((init_train, init_train[:6000], init_train[6000:]))


x_val = np.load(path + 'x_val.npy')
y_val = np.load(path + 'y_val.npy')
mask_val = np.load(path + 'mask_val.npy')[:, :, 0, :]
mask_val_img = np.repeat(np.expand_dims(mask_val, 2), n_img, 2)
mask_val_out = np.repeat(np.expand_dims(mask_val, 2), n_output, 2)
init_val = np.load(path + 'init_val.npy')
x_val = np.concatenate((x_val, getNoise(x_val[:2113], 40), getNoise(x_val[2113:], 50)))
y_val = np.concatenate((y_val, y_val[:2113], y_val[2113:]))
mask_val = np.concatenate((mask_val, mask_val[:2113], mask_val[2113:]))
mask_val_img = np.concatenate((mask_val_img, mask_val_img[:2113], mask_val_img[2113:]))
mask_val_out = np.concatenate((mask_val_out, mask_val_out[:2113], mask_val_out[2113:]))
init_val = np.concatenate((init_val, init_val[:2113], init_val[2113:]))

x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')
mask_test = np.load(path + 'mask_test.npy')[:, :, 0, :]
mask_test_img = np.repeat(np.expand_dims(mask_test, 2), n_img, 2)
mask_test_out = np.repeat(np.expand_dims(mask_test, 2), n_output, 2)
init_test = np.load(path + 'init_test.npy')


# x_train [12678, 208] x_val[4226, 208] x_test[4231, 208]

# normalization
# scaler1 = Normalizer().fit(x_train)
# x_train = scaler1.transform(x_train)
# x_val = scaler1.transform(x_val)
# x_test = scaler1.transform(x_test)


# Create the dataset
class SimDataset(Dataset):
    def __init__(self, voltages, conductivity, mask_img, mask_out, initial, transform=None):
        self.voltages, self.conductivity, self.mask_img, self.mask_out, self.initial = voltages, conductivity, \
                                                                                      mask_img, mask_out, initial
        self.transform = transform

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, idx):
        voltage = self.voltages[idx].reshape(208, 1)  # bz, 208, 1
        conduct = self.conductivity[idx]
        mask_img = self.mask_img[idx]
        mask_out = self.mask_out[idx]
        initial = self.initial[idx]
        return [voltage, conduct, mask_img, mask_out, initial]


trans = transforms.Compose([
    transforms.ToTensor(),
])


batch_size = 32

datasets = {
    "train": SimDataset(x_train, y_train, mask_train_img, mask_train_out, init_train, transform=trans),
    "valid": SimDataset(x_val, y_val, mask_val_img, mask_val_out, init_val, transform=trans),
    "test": SimDataset(x_test, y_test, mask_test_img, mask_test_out, init_test, transform=trans)
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

# calculate the float operation points to evaluate the complexity
# flops = FlopCountAnalysis(model, inputs=(torch.randn((32, 208, 1)), torch.randn((32, 3, 200, 3)), 32,))
# print(flop_count_table(flops))

model = model.to(device)

initiala = marsagliasample2(n_output, 0.2)
initialb = marsagliasample2(n_output, 0.1)
initialc = marsagliasample2(n_output, 0.05)


def CD_loss(data_p, data_t):
    """
    Arg:
        data_p: prediction of point clouds [N,A,L,D]
        (N: batchsize, A:amount of objects, L:length of objects, D:dimensions)
        data_t: ground truth of point clouds

    Return: mean CD_loss between data_p and data_t
    """
    N = data_p.size()[0]
    A = data_p.size()[1]
    L = data_p.size()[2]
    D = data_p.size()[3]
    data_p = data_p.permute(2, 0, 1, 3)  # [L, N, A, D]
    data_t = data_t.permute(2, 0, 1, 3)
    data_p1 = data_p.unsqueeze(1).expand(L, L, N, A, D)
    data_t1 = data_t.unsqueeze(1).expand(L, L, N, A, D)
    d_p2t = torch.min(torch.mean((data_p1 - data_t)**2, dim=-1), dim=1)[0]
    d_t2p = torch.min(torch.mean((data_t1 - data_p)**2, dim=-1), dim=1)[0]

    return torch.sum(d_p2t + d_t2p) / (D * N * L * A)


# different length Chamfer distance loss function
class dl_CD_loss(nn.Module):

    def __init__(self):
        super(dl_CD_loss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P, bs, points_dim, num_points_x, num_points_y = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return torch.sum(loss_1/(bs*points_dim*num_points_x) + loss_2/(bs*points_dim*num_points_y))

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P, bs, points_dim, num_points_x, num_points_y


loss_fn = CD_loss


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


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
            vols, imgs, mask_img, mask_out, initial = data
            vols = vols.permute(1, 0, 2).float().to(device) # 1, bz, 208
            imgs = imgs.float().to(device)
            mask_img = mask_img.float().to(device)
            mask_out = mask_out.float().to(device)
            initial1 = torch.from_numpy(initiala).expand(vols.size(1), n_output, 3).float().to(device)
            initial2 = torch.from_numpy(initialb).expand(vols.size(1), n_output, 3).float().to(device)
            initial3 = torch.from_numpy(initialc).expand(vols.size(1), n_output, 3).float().to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model.forward(sensor_pos, vols, initial1, initial2, initial3,
                                                      batch_size=vols.size(1))
            output = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1), output3.unsqueeze(1)], 1)
            imgs = mask_img*imgs
            output = mask_out*output
            loss = loss_fn(imgs, output)
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
                vols, imgs, mask_img, mask_out, initial = data
                vols = vols.permute(1, 0, 2).float().to(device)
                imgs = imgs.float().to(device)
                mask_img = mask_img.float().to(device)
                mask_out = mask_out.float().to(device)
                initial1 = torch.from_numpy(initiala).expand(vols.size(1), n_output, 3).float().to(device)
                initial2 = torch.from_numpy(initialb).expand(vols.size(1), n_output, 3).float().to(device)
                initial3 = torch.from_numpy(initialc).expand(vols.size(1), n_output, 3).float().to(device)
                output1, output2, output3 = model.forward(sensor_pos, vols, initial1, initial2, initial3,
                                                          batch_size=vols.size(1))
                output = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1), output3.unsqueeze(1)], 1)
                imgs = mask_img*imgs
                output = mask_out*output
                # loss = loss_fn(imgs.reshape(vols.size(1)*3, n_img, 3), output.reshape(vols.size(1)*3, n_output, 3))
                loss = loss_fn(imgs, output)
                vali_loss += loss.item() * vols.size(1)
                epoch_samples += vols.size(1)

                tepoch.set_postfix(ValidLoss=vali_loss / epoch_samples)

        val_loss_plot.append(vali_loss / epoch_samples)
        #
        # if train_lossall < best_val_loss:
        #     best_val_loss = train_lossall
        #     torch.save(model.state_dict(), './saved_Models/new_shape_model.pt')
        #     sleep(0.1)

        # np.save('./loss_curve/new_train_loss.npy', train_loss_plot)
        # np.save('./loss_curve/new_val_loss.npy', val_loss_plot)

print('training end')