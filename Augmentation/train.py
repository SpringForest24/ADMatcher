from .utils import *
from .model import *
from .evaluate import *

def train(gconv, contrast_model, dataloader, optimizer, device, pn, args):
    gconv.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        if args.core == 'edrop':
            x1, edge_index1, batch1 = drop_edge(data, pn)#删除边
        elif args.core == 'ndrop':
            x1, edge_index1, batch1 = drop_node(data, pn)#删除节点
        elif args.core == 'fmask':
            x1, edge_index1, batch1 = mask_nodes(data, pn)#特征掩盖
        elif args.core == 'subsample':
            x1, edge_index1, batch1 = bfs_subgraph(data, pn)#bfs子图采集
        elif args.core == 'no':
            x1, edge_index1, batch1 = data.x, data.edge_index, data.batch #不进行Data Augmentation
        else:
            break
        _, g1 = gconv(data.x, data.edge_index, data.batch)
        _, g2 = gconv(x1, edge_index1, batch1)
        

        g1, g2 = [gconv.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def train1(gconv, contrast_model, dataloader, optimizer, device, pn, args):#随机自动增强,调用了mix_aug
    gconv.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        if args.core == 'random':
            x1, edge_index1, batch1 = mix_aug(data, pn, dataloader, args)
        else:
            break
        _, g1 = gconv(data.x, data.edge_index, data.batch)
        _, g2 = gconv(x1, edge_index1, batch1)
        
        g1, g2 = [gconv.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def train2(gconv, contrast_model, dataloader, optimizer, device, getcore):#使用K-core和k-truss增强
    gconv.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)

        data = data.to(device)
        _, g1 = gconv(data.x, data.edge_index, data.batch)
        _, g2 = gconv(*getcore.drop_node(data))#使用的是k_core子图节点保留，其他节点有概率被drop

        g1, g2 = [gconv.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss