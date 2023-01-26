import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import pandas as pd
import gc
import scipy.io as io
import torch
from torch import device, nn, reshape, functional, softmax
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from shape_model import Model
from pos_model import Model as Model_pos
from conduct_model import Model as Model_cdt
from function import marsagliasample2, getNoise, plotloss, switchmeasurements
import matlab.engine


# empty the cache
torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# import data
path = './data/random_500/'
n_img = 500
SNR = 100

x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')
mask_test = np.load(path + 'mask_test.npy')[:, :, 0, :]
mask_test_shape = np.repeat(np.expand_dims(mask_test, 2), n_img, 2)
pos_origin = np.load(path + 'pos_test.npy')
pos_test = np.load(path + 'pos_test.npy')
pos_test[:, :, 2] = pos_test[:, :, 2]/15
pos_test[:, :, :2] = pos_test[:, :, :2]/(11.48*2)
cdt_test = np.load(path + 'cdt_test.npy')
cdt_test[np.where(cdt_test == 0.1)] = 1
cdt_test[np.where(cdt_test == 3.9)] = 2

path2 = "./data/"
sensor_pos = np.load(path2 + 'sensor_208.npy').reshape(208, 1, 6)
sensor_pos = torch.from_numpy(sensor_pos).float().to(device)  # [208, 1, 6]

initiala = marsagliasample2(n_img, 0.2)
initialb = marsagliasample2(n_img, 0.1)
initialc = marsagliasample2(n_img, 0.05)

# add noise to the testing data
if SNR != 100:
    x_test = getNoise(x_test, SNR)


# create the dataset
class SimDataset(Dataset):
    def __init__(self, voltages, voltages2, conductivity, pos, cdt, mask_shape, mask_pos, transform=None):
        self.voltages, self.voltages2, self.conductivity, self.pos, self.cdt, self.mask_shape, self.mask_pos = \
            voltages, voltages2, conductivity, \
                                                                                     pos, cdt, mask_shape, mask_pos
        self.transform = transform

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, idx):
        voltage = self.voltages[idx].reshape(208, 1)  # bz, 1, 208
        voltage2 = self.voltages2[idx].reshape(1, 208)
        conduct = self.conductivity[idx]
        pos = self.pos[idx]
        cdt = self.cdt[idx]
        mask_shape = self.mask_shape[idx]
        mask_pos = self.mask_pos[idx]
        return [voltage, voltage2, conduct, pos, cdt, mask_shape, mask_pos]


trans = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 32

datasets = {
    "test": SimDataset(x_test, x_test, y_test, pos_origin, cdt_test, mask_test_shape, mask_test, transform=trans)
}

dataloaders = {
    "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
}

# define hyperparameters
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

# load models
model = Model(n_point=n_point, n_embed=n_embed, n_hid=n_hid, n_head=n_head, n_elayers=n_elayers, n_mlayers=n_mlayers,
              n_dlayers= n_dlayers, dropout=dropout
              ).to(device)

model_pos = Model_pos(n_point=n_point, n_embed=n_embed, n_hid=n_hid, n_head=n_head, n_elayers=n_elayers, n_mlayers=n_mlayers,
              n_dlayers= n_dlayers, dropout=dropout
              ).to(device)

model_cdt = Model_cdt(n_point=n_point, n_embed=n_embed, n_hid=n_hid, n_head=n_head, n_elayers=n_elayers, n_mlayers=n_mlayers,
              n_dlayers= n_dlayers, dropout=dropout
              ).to(device)

model.load_state_dict(torch.load('./saved_Models/0305_3Dshape_model.pt'))
model_pos.load_state_dict(torch.load('./saved_Models/0315_pos_model.pt'))
model_cdt.load_state_dict(torch.load('./saved_Models/0308_cdt_model.pt'))
model.eval()


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


def HD_loss(data_p, data_t):
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
    d_p2t_max = torch.max(d_p2t)
    d_t2p_max = torch.max(d_t2p)

    return torch.max(d_t2p_max, d_p2t_max)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


loss_fn_CD = CD_loss
loss_fn_HD = HD_loss
loss_fn_MSE = nn.MSELoss()
loss_fn_cross = nn.CrossEntropyLoss()

# randomly select a sample from testing dataset and plot
a = np.random.randint(0, len(x_test), 1).item()
# a = 3950
vols = torch.from_numpy(x_test[a].reshape(-1, 1, 1)).float().to(device)
vols_pos = torch.from_numpy(x_test[a].reshape(1, 1, -1)).float().to(device)
pos_gt = torch.from_numpy(pos_origin[a]).float().to(device)
shape_gt = torch.from_numpy(y_test[a]*11.78).float().to(device)
mask_pos = torch.from_numpy(mask_test[a]).float().to(device)
initial1 = torch.from_numpy(initiala).expand(vols.size(1), n_img, 3).float().to(device)
initial2 = torch.from_numpy(initialb).expand(vols.size(1), n_img, 3).float().to(device)
initial3 = torch.from_numpy(initialc).expand(vols.size(1), n_img, 3).float().to(device)
mask_shape = torch.from_numpy(mask_test_shape[a]).float().to(device)
pos_pre = model_pos.forward(sensor_pos, vols_pos, batch_size=vols.size(1))[0]*mask_pos
pos_pre[:, 2] = pos_pre[:, 2]*15
pos_pre[:, :2] = pos_pre[:, :2]*(11.48*2)
output1, output2, output3 = model.forward(sensor_pos, vols, initial1, initial2, initial3, batch_size=vols.size(1))
output = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1), output3.unsqueeze(1)], 1)[0]*11.78*mask_shape
pos_pre = pos_pre.unsqueeze(1).expand(3, n_img, 3)
pos_gt = pos_gt.unsqueeze(1).expand(3, n_img, 3)

gt = shape_gt + pos_gt
reconstruction = output + pos_pre
loss_CD = loss_fn_CD(gt.unsqueeze(0), reconstruction.unsqueeze(0))
loss_HD = loss_fn_HD(gt.unsqueeze(0), reconstruction.unsqueeze(0))
gt = np.array(gt.detach().cpu())
reconstruction = np.array(reconstruction.detach().cpu())

cdt_predict = model_cdt.forward(sensor_pos, vols_pos, batch_size=vols.size(1)).reshape(-1, 3)
cdt_gt = torch.from_numpy(cdt_test[a]).long().to(device).reshape(-1)
loss_cross = loss_fn_cross(cdt_predict, cdt_gt)
acc = accuracy(cdt_predict, cdt_gt)/len(cdt_gt)
cdt_predict = cdt_predict.argmax(axis=1)
cdt_gt = np.array(cdt_gt.detach().cpu())
cdt_predict = np.array(cdt_predict.detach().cpu())
loss_CD = np.array(loss_CD.detach().cpu())
loss_HD = np.array(loss_HD.detach().cpu())
loss_cross = np.array(loss_cross.detach().cpu())
# acc = np.array(acc.detach().cpu())

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(gt[0].T[0], gt[0].T[1], gt[0].T[2])
ax.scatter(gt[1].T[0], gt[1].T[1], gt[1].T[2])
ax.scatter(gt[2].T[0], gt[2].T[1], gt[2].T[2])
plt.title(f'{a}'+' Ground Truth')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax.set_zlim([0, 20])

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(reconstruction[0].T[0], reconstruction[0].T[1], reconstruction[0].T[2])
ax.scatter(reconstruction[1].T[0], reconstruction[1].T[1], reconstruction[1].T[2])
ax.scatter(reconstruction[2].T[0], reconstruction[2].T[1], reconstruction[2].T[2])
plt.title(f'{a}'+' Reconstruction')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax.set_zlim([0, 20])
plt.show()

#the default reshape order in numpy is C order, need to change it to the matlab default order, which is fortran order
gt = np.reshape(gt, (3*n_img, 3), order="F")
reconstruction = np.reshape(reconstruction, (3*n_img, 3), order="F")
infodata = {'num': a, 'SNR': SNR, 'CD':loss_CD, 'HD': loss_HD, 'crossentropy':loss_cross, 'accuracy': acc,
            'gt': cdt_gt, 'pre': cdt_predict}
dfinfo = DataFrame(infodata)
dfinfo.to_csv('./matplot/information.csv')
np.savetxt('./matplot/sim_gt.txt', gt)
np.savetxt('./matplot/sim_pre.txt', reconstruction)
np.savetxt('./matplot/sim_vols.txt', x_test[a])

# Evaluate on the whole simulation test dataset
vali_CD = 0
vali_HD = 0
vali_Cross = 0
vali_acc = 0
vali_pos = 0
epoch_samples = 0
best_val_loss = float('inf')
i = 0
model.eval()
with torch.no_grad():
    with tqdm(dataloaders['test'], unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {i}/{epoch}")
            vols, vols2, imgs, pos, cdt, mask_shape, mask_pos = data
            vols = vols.permute(1, 0, 2).float().to(device)
            vols2 = vols2.permute(1, 0, 2).float().to(device)
            imgs = imgs.float().to(device)*11.78
            pos = pos.float().to(device)
            cdt = cdt.long().to(device).reshape(-1)
            mask_shape = mask_shape.float().to(device)
            mask_pos = mask_pos.float().to(device)
            initial1 = torch.from_numpy(initiala).expand(vols.size(1), n_img, 3).float().to(device)
            initial2 = torch.from_numpy(initialb).expand(vols.size(1), n_img, 3).float().to(device)
            initial3 = torch.from_numpy(initialc).expand(vols.size(1), n_img, 3).float().to(device)
            output1, output2, output3 = model.forward(sensor_pos, vols, initial1, initial2, initial3, batch_size=vols.size(1))
            output = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1), output3.unsqueeze(1)], 1)*11.78*mask_shape
            # imgs = mask_shape * imgs*11.78
            # output = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1), output3.unsqueeze(1)],
            #                    1) * mask_shape
            pos_pre = model_pos.forward(sensor_pos, vols2, batch_size=vols.size(1))*mask_pos
            pos_pre[:, 2] = pos_pre[:, 2] * 15
            pos_pre[:, :2] = pos_pre[:, :2] * (11.48 * 2)
            pos_pre = pos_pre.unsqueeze(2).expand(vols.size(1), 3, n_img, 3)
            pos = pos.unsqueeze(2).expand(vols.size(1), 3, n_img, 3)
            gt2 = imgs + pos
            reconstruction2 = output + pos_pre
            # loss_CD1 = loss_fn_CD(imgs, output)
            # loss_CD2 = loss_fn_CD(pos, pos_pre)
            loss_mse = loss_fn_MSE(pos, pos_pre)
            # loss_CD = loss_fn_CD(gt2, reconstruction2)
            # loss_HD = loss_fn_HD(gt2, reconstruction2)
            loss_CD = loss_fn_CD(imgs, output)
            loss_HD = loss_fn_HD(imgs, output)
            cdt_predict = model_cdt.forward(sensor_pos, vols2, batch_size=vols.size(1)).reshape(-1, 3)
            loss_cross = loss_fn_cross(cdt_predict, cdt)
            acc = accuracy(cdt_predict, cdt)/len(cdt)

            vali_CD += loss_CD.item()*vols.size(1)
            vali_HD += loss_HD.item()*vols.size(1)
            vali_Cross += loss_cross.item()
            vali_pos += loss_mse.item()
            epoch_samples += vols.size(1)

            tepoch.set_postfix(TestCD=vali_CD / epoch_samples, TestHD=vali_HD / epoch_samples, TestCross=vali_Cross /
                                                                                                      epoch_samples,
                               Testacc = acc, Test_pos = vali_pos/epoch_samples)






