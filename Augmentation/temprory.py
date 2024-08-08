from utils import Synthetic_Dataset

#测试一下Synthetic dataset类
dataset = Synthetic_Dataset(root='../data/synthetic_data')
print(len(dataset))
print(dataset[0])
