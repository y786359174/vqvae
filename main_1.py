import torch
import torch.nn as nn
import torch.nn.functional as F
from vqvae import VQVAE
from pixel_embed import PixelCNNWithEmbedding
from dataset import get_dataloader, get_img_shape
import time
import cv2
import einops
import os
import numpy as np
## vae training

def train_vqvae(model: VQVAE,
                img_shape=None,
                device='cuda',
                ckpt_path='dldemos/VQVAE/model.pth',
                batch_size=64,
                dataset_type='MNIST',
                lr=1e-3,
                n_epochs=100,
                l_w_embedding=1,
                l_w_commitment=0.25):
    print('batch size:', batch_size)
    # dataloader = get_dataloader(dataset_type,
    #                             batch_size,
    #                             img_shape=img_shape,
    #                             use_lmdb=USE_LMDB)
    dataloader = get_dataloader(batch_size, num_workers=4)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x,_ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        reconstruct(vqvae, device)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def reconstruct(model, device, dataset_type='MNIST'):
    # dataloader = get_dataloader(cfg['dataset_type'],
    #                             16,
    #                             img_shape=(img_shape[1], img_shape[2]))
    dataloader = get_dataloader(batch_size = 16, num_workers=4)
    img, _ = next(iter(dataloader))
    x = img.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    # x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    x_cat = (x_cat.clip(-1, 1) + 1) / 2 * 255
    x_cat = x_cat.cpu().numpy().astype(np.uint8)
    # print(x_new.shape)
    if(x_cat.shape[2]==3):
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'data/anime_faces64/vqvae_reconstruct_{dataset_type}.jpg', x_cat)

## pixelcnn training

def train_generative_model(vqvae: VQVAE,
                           model,
                           img_shape=None,
                           device='cuda',
                           ckpt_path='dldemos/VQVAE/gen_model.pth',
                           dataset_type='MNIST',
                           batch_size=64,
                           n_epochs=50):
    print('batch size:', batch_size)
    # dataloader = get_dataloader(dataset_type,
    #                             batch_size,
    #                             img_shape=img_shape,
    #                             use_lmdb=USE_LMDB)
    dataloader = get_dataloader(batch_size, num_workers=4)
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x,_ in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)

            predict_x = model(x)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')

## sampling

def sample_imgs(vqvae: VQVAE,
                gen_model,
                img_shape,
                n_sample=25,
                device='cuda',
                dataset_type='MNIST'):
    vqvae = vqvae.to(device)
    vqvae.eval()
    gen_model = gen_model.to(device)
    gen_model.eval()

    C, H, W = img_shape
    H, W = vqvae.get_latent_HW((C, H, W))
    input_shape = (n_sample, H, W)
    x = torch.zeros(input_shape).to(device).to(torch.long)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = gen_model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1)
                x[:, i, j] = pixel[:, 0]

    x_cat = vqvae.decode(x)

    x_cat = einops.rearrange(x_cat,
                            '(n1 n2) c h w -> (n1 h) (n2 w) c',
                            n1=int(n_sample**0.5))

    # imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    x_cat = (x_cat.clip(-1, 1) + 1) / 2 * 255
    x_cat = x_cat.cpu().numpy().astype(np.uint8)
    # print(x_new.shape)
    if(x_cat.shape[2]==3):
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'data/anime_faces64/vqvae_sample_{dataset_type}.jpg', x_cat)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', type=int, default=0)
    # parser.add_argument('-d', type=int, default=0)
    # args = parser.parse_args()
    # cfg = get_cfg(args.c)
    cfg = dict( dataset_type='CelebAHQ',
                img_shape=get_img_shape(),
                dim=128,
                n_embedding=64,
                batch_size=64,
                n_epochs=200,
                l_w_embedding=1,
                l_w_commitment=0.25,
                lr=2e-4,
                n_epochs_2=200,
                batch_size_2=32,
                pixelcnn_n_blocks=15,
                pixelcnn_dim=384,
                pixelcnn_linear_dim=256,
                vqvae_path='data/anime_faces64/model_vqvae_vqvae.pth',
                gen_model_path='data/anime_faces64/pixelcnn.pth')

    device = 'cuda:0'

    img_shape = cfg['img_shape']

    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding'])
    gen_model = PixelCNNWithEmbedding(cfg['pixelcnn_n_blocks'],
                                      cfg['pixelcnn_dim'],
                                      cfg['pixelcnn_linear_dim'], True,
                                      cfg['n_embedding'])
    # 1. Train VQVAE
    train_vqvae(vqvae,
                img_shape=(img_shape[1], img_shape[2]),
                device=device,
                ckpt_path=cfg['vqvae_path'],
                batch_size=cfg['batch_size'],
                dataset_type=cfg['dataset_type'],
                lr=cfg['lr'],
                n_epochs=cfg['n_epochs'],
                l_w_embedding=cfg['l_w_embedding'],
                l_w_commitment=cfg['l_w_commitment'])

    # 2. Test VQVAE by visualizaing reconstruction result
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
    # # dataloader = get_dataloader(cfg['dataset_type'],
    # #                             16,
    # #                             img_shape=(img_shape[1], img_shape[2]))
    # dataloader = get_dataloader(batch_size = 16, num_workers=4)
    # img, _ = next(iter(dataloader))
    # img = img.to(device)
    reconstruct(vqvae, device)

    # 3. Train Generative model (Gated PixelCNN in our project)
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))

    train_generative_model(vqvae,
                           gen_model,
                           img_shape=(img_shape[1], img_shape[2]),
                           device=device,
                           ckpt_path=cfg['gen_model_path'],
                           dataset_type=cfg['dataset_type'],
                           batch_size=cfg['batch_size_2'],
                           n_epochs=cfg['n_epochs_2'])

    # 4. Sample VQVAE
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
    gen_model.load_state_dict(torch.load(cfg['gen_model_path']))
    sample_imgs(vqvae,
                gen_model,
                cfg['img_shape'],
                device=device,
                dataset_type=cfg['dataset_type'])