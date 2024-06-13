import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from vqvae import VQVAE
from pixel_embed import PixelCNNWithEmbedding
sys.path.append('../')
from CycleGAN_own.dataset import get_dataloader, get_img_shape
import time
import cv2
import einops
import os
import numpy as np

## vae training

def train_vqvae(vqvae: VQVAE,
                ckpt_path,
                device='cuda'):
    batch_size=256
    lr=2e-4
    n_epochs=10000
    l_w_embedding=1
    l_w_commitment=0.25

    mse_loss = nn.MSELoss()
    Img_dir_name = '../face2face/dataset/train/C'
    dataloader = get_dataloader(batch_size, data_dir=Img_dir_name, num_workers=4)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr)
    vqvae.to(device)
    vqvae.train()

    for epoch_i in range(n_epochs):
        tic = time.time()
        total_loss = 0
        for x,_ in dataloader:
            current_batch_size = x.shape[0]

            x = x.to(device)
            x_hat, ze, zq = vqvae(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)     # 让zq去接近ze
            l_commitment = mse_loss(ze, zq.detach())    # 让ze去接近zq
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment # 让zq多接近ze，码本多贴合真实数据
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()    
        
        if(epoch_i%20==0):
            torch.save(vqvae.state_dict(), ckpt_path)
            reconstruct(vqvae, device)
            print(f'epoch {epoch_i} total_loss {total_loss}  time: {(toc - tic):.2f}s')
            print(f'recons_loss {l_reconstruct:.4e} embed_loss {l_w_embedding * l_embedding:.4e} commit_loss {l_w_commitment * l_commitment:.4e}')

recon_time = 0
def reconstruct(vqvae:VQVAE, device='cuda'):
    global recon_time
    recon_time += 1
    i_n = 5
    # for i in range(i_n*i_n):
    vqvae = vqvae.to(device)
    vqvae = vqvae.eval()
    Img_dir_name = '../face2face/dataset/train/C'
    dataloader = get_dataloader(i_n * i_n, data_dir=Img_dir_name, num_workers=4)
    x_real, _ = next(iter(dataloader))
    x_real = x_real.to(device)
    with torch.no_grad():
        x_new, _ , _  = vqvae(x_real)
        x_new = x_new.detach().cpu()
        x_real = x_real.cpu()
        x_cat = torch.concat((x_real, x_new), 3)
        x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        x_cat = (x_cat.clip(-1, 1) + 1) / 2 * 255
        x_cat = x_cat.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_cat.shape[2]==3):
            x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir,'./vqvae_reconstruct_%d.jpg' % (recon_time)), x_cat)
    vqvae = vqvae.train()


## pixelcnn training

def train_pixelcnn( vqvae: VQVAE,
                    pixelcnn: PixelCNNWithEmbedding,
                    ckpt_path,
                    device='cuda'):
    batch_size = 256
    lr=2e-4
    n_epochs = 1000
    
    loss_fn = nn.CrossEntropyLoss()
    dataloader = get_dataloader(batch_size, num_workers=4)
    optimizer = torch.optim.Adam(pixelcnn.parameters(), lr)
    vqvae.to(device)
    vqvae.eval()
    pixelcnn.to(device)
    pixelcnn.train()

    for epoch_i in range(n_epochs):
        tic = time.time()
        total_loss = 0
        for x,_ in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():           # 上面eval好像仍然会算梯度，耽误时间
                x = x.to(device)
                zq_index = vqvae.encode(x)  # 生成最近邻zq的index
            index_gen = pixelcnn(zq_index)
            loss = loss_fn(index_gen, zq_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()    
        
        if(epoch_i%20==0):
            torch.save(pixelcnn.state_dict(), ckpt_path)
            # reconstruct(vqvae, device)
            sample(vqvae, pixelcnn, device)
            print(f'epoch {epoch_i} total_loss {total_loss}  time: {(toc - tic):.2f}s')
            
sample_time = 0
def sample(vqvae:VQVAE, pixelcnn: PixelCNNWithEmbedding, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 5
    # for i in range(i_n*i_n):
    vqvae = vqvae.to(device)
    vqvae = vqvae.eval()
    pixelcnn = pixelcnn.to(device)
    pixelcnn = pixelcnn.eval()
    C, H, W = get_img_shape()
    H, W = vqvae.get_latent_HW((C, H, W))       # 这个就是我们要生成的大小了
    input_shape = (i_n*i_n, H, W)
    zq_gen = torch.zeros(input_shape).to(device).to(torch.long)  
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = pixelcnn(zq_gen)
                prob_dist = F.softmax(output[:, :, i, j], dim=-1)   #dim=-1会对最后一个维度softmax，也就是这里的Channal维
                pixel = torch.multinomial(prob_dist, 1)             #概率采样，采1个
                zq_gen[:, i, j] = pixel[:, 0]                       # pixel是[25,1]
        
        x_new = vqvae.decode(zq_gen) #zq_gen只是生成的zq，还要过decode
        x_new = x_new.detach().cpu()
        x_new = einops.rearrange(x_new, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir,'./vqvae_sample_%d.jpg' % (sample_time)), x_new)
    vqvae = vqvae.train()



## 定义参数初始化函数
def weights_init_normal(m):                                    
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字. 
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("Linear") != -1:  # 添加线性层的初始化
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

save_dir = './data/selfish_faces64'

if __name__ == '__main__':

    vqvae_ckpt_path = os.path.join(save_dir,'model_vqvae.pth') 
    pixelcnn_ckpt_path = os.path.join(save_dir,'model_pixelcnn.pth') 
    device = 'cuda'
    image_shape = get_img_shape()
    n_embedding = 128
    vqvae = VQVAE(image_shape[0], dim = 256, n_embedding = n_embedding).to(device)
    pixelcnn = PixelCNNWithEmbedding(   n_blocks=15,
                                        p = 384,
                                        linear_dim = 256,
                                        bn = True,
                                        color_level = n_embedding)
    vqvae.apply(weights_init_normal)
    pixelcnn.apply(weights_init_normal)
    # vqvae.load_state_dict(torch.load(vqvae_ckpt_path))
    # pixelcnn.load_state_dict(torch.load(pixelcnn_ckpt_path))

    train_vqvae(vqvae, vqvae_ckpt_path, device)
    # train_pixelcnn(vqvae, pixelcnn, pixelcnn_ckpt_path, device)


    # vqvae.load_state_dict(torch.load(vqvae_ckpt_path))
    # reconstruct(vqvae, device=device)
    # pixelcnn.load_state_dict(torch.load(pixelcnn_ckpt_path))
    # sample(vqvae, pixelcnn, device)


