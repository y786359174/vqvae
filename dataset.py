import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor, Pad, Resize
import einops
import numpy as np
import cv2



# def get_dataloader(batch_size: int):
#     transform = Compose([ToTensor(), 
#                          Pad(padding=2, fill=0),
#                          Lambda(lambda x: (x - 0.5) * 2)
#                          ])
#     dataset = torchvision.datasets.MNIST(root='./data/mnist',
#                                          transform=transform, download=True)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# def get_img_shape():
#     return (1, 32, 32)

def get_dataloader(batch_size: int, num_workers = 0):
    transform = Compose([ToTensor(), 
                         Resize(64),
                         Lambda(lambda x: (x - 0.5) * 2)])
    data_dir = '../faces'
    dataset = torchvision.datasets.ImageFolder(root=data_dir,
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

def get_img_shape():
    # return (3, 96, 96)
    return (3, 64, 64)

def module_test():
    batch_size = 64
    dataloader = get_dataloader(batch_size)
    data_iter = iter(dataloader)                # 测试，只提取一次dataloader，把它放入迭代器，next读取一个
    x,_ = next(data_iter)
    image_shape = x.shape
    print("x.shape =",image_shape)
    x = (x + 1) / 2 * 255
    x = x.clamp(0, 255)
    x_new = einops.rearrange(x,                                     # 新库，对张量维度重新变化
                            '(b1 b2) c h w -> (b1 h) (b2 w) c',     # 还可以自己分解和重组，括号里的就是分解/重组的部分
                            b1=int(image_shape[0]**0.5))
    x_new = x_new.numpy().astype(np.uint8)
    print(x_new.shape)
    if(x_new.shape[2]==3):
        x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
    cv2.imwrite('minst88.jpg', x_new)
    


if __name__ == '__main__':
    module_test()