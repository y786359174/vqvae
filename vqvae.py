import torch
import torch.nn as nn


class ResidualBlock(nn.Module):     # 很简单的残差层，有一个3*3卷积和一个1*1卷积

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp
    
class VQVAE(nn.Module):

    def __init__(self, input_dim, dim, n_embedding):                            # dim是encoder后的编码向量的长度c
        super().__init__()                                                      # hdim同时也是我们维护的码本的向量长度c
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), 
                                     nn.Conv2d(dim, dim, 4, 2, 1),              # 421 每过一次下采样一次，这里一共下采样两次
                                     nn.ReLU(), 
                                     nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), 
                                     ResidualBlock(dim))
        # vq_codebook = nn.Parameter(torch.empty(k, hidden_c))                      # 码本，k个长度c的向量，最近邻离散时只是用了这个矩阵
        self.vq_embedding = nn.Embedding(n_embedding, dim)                          # 这里n_embedding就是码本里的向量个数K，也就是离散域的大小
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)                   # 使用均匀分布初始化码本
                                                                                    # 为什么要用embedding层做码本呢？因为在把离散index值转成连续的向量时，可以直接套这个层，在之后的decode里可以看到这个用法
        self.decoder = nn.Sequential(   
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), 
            ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),                              # 转置卷积就是上采样，同样也是两次
            nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2   # 这对上采样次数做了一个没用的标注, 后面会用到

    def forward(self, x):
        # encode
        ze = self.encoder(x)    # encoder后的z，记为ze，连续z
        
        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data       # 码本codebook，k*c，k个向量，长度是c的向量，所有ze都要贴近这个向量取值
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)    # 传播计算得到每个向量对每个码本向量的距离。很关键
        nearest_neighbor = torch.argmin(distance, 1)                        # 得到最贴近的离散向量的index
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)        # 此时ze转换成了zq，这里用的embedding层就是个映射，一个值（index）映射成一个向量（离散的）
                                                                            # 0312只是把索引顺序改成BHWC
        # stop gradient                                                     
        decoder_input = ze + (zq - ze).detach()
        
        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq        # 这是训练vqvae用的，生成用不了这么多东西
    
    @torch.no_grad()                # pixelcnn通过生成码本index值作为生成图片的输入，因此需要个训练的模块学习其分布吧，所以这里把最近邻离散及之前的拆出来了供pixelcnn学他的分布
    def encode(self, x):            # 为啥关掉梯度传播？可能是不想训练pixelcnn时改变vqvae的网络值吧
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat
    
                                       # pixelcnn通过生成码本index值作为生成图片的输入，因此需要个训练的模块学习其分布吧，所以这里把最近邻离散及之前的拆出来了供pixelcnn学他的分布
    def encode_ze(self, x):            # 为啥关掉梯度传播？可能是不想训练pixelcnn时改变vqvae的网络值吧
        ze = self.encoder(x)
        return ze

    
    def decode_ze(self, ze):
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)