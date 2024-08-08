from torch_geometric.datasets import TUDataset, PPI, QM9
from torch_geometric.data import Data, Dataset
import torch_geometric.utils as utils
from torch_geometric.loader import DataLoader
import torch
import time
import networkx as nx
import numpy as np
from tqdm import tqdm
import random
import signal
#from vis import plot_graph, plot_pair, plot_graph1, plot_aligned#后续再加进来
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing
from torch_geometric.utils import degree
import torch.nn.functional as F
from Augmentation.utils import Synthetic_Dataset

#如果节点特征存在则使用节点特征，如果节点特征不存在，则使用[num_node,3]的全1向量作为特征
def load_dataset(dataset_name):
    if dataset_name == 'ENZYMES':
        dataset = TUDataset(root='./datasets/ENZYMES', name='ENZYMES')
    elif dataset_name == 'MUTAG':
        dataset = TUDataset(root='./datasets/MUTAG', name='MUTAG')
    elif dataset_name == 'PROTEINS':
        dataset = TUDataset(root='./datasets/PROTEINS', name='PROTEINS')
    elif dataset_name == 'REDDIT-BINARY':
        dataset = TUDataset(root='./datasets/REDDIT-BINARY', name='REDDIT-BINARY')
    elif dataset_name == 'COX2':
        dataset = TUDataset(root='./datasets/COX2', name='COX2')
    elif dataset_name == 'AIDS':
        dataset = TUDataset(root='./datasets/AIDS', name='AIDS')
    elif dataset_name == 'NCI1':
        dataset = TUDataset(root='./datasets/NCI1', name='NCI1')
    elif dataset_name == 'DD':
        dataset = TUDataset(root='./datasets/DD', name='DD')
    elif dataset_name == 'COLLAB':
        dataset = TUDataset(root='./datasets/COLLAB', name='COLLAB')
    elif dataset_name == 'IMDB-BINARY':
        dataset = TUDataset(root='./datasets/IMDB-BINARY', name='IMDB-BINARY')
    elif dataset_name == 'PPI':
        dataset = PPI(root='./datasets/PPI')
    elif dataset_name == 'QM9':
        dataset = QM9(root='./datasets/QM9')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    
    dataset_with_id = []
    if dataset[0].x is None:
        # 如果节点特征不存在，使用get_dataset_one方法生成全1特征
        for i in range(len(dataset)):
            data = dataset[i]
            data.idx = i
            data.x = torch.ones((data.num_nodes, 3)).float()#全1
            # data.x = torch.randn((data.num_nodes, 5)).float()#人为制造一个节点特征
            dataset_with_id.append(data)
        return dataset_with_id
    else:
        return dataset

#使用的是节点的度作为特征
def load_dataset2(dataset_name):
    if dataset_name == 'ENZYMES':
        dataset = TUDataset(root='./datasets/ENZYMES', name='ENZYMES')
    elif dataset_name == 'MUTAG':
        dataset = TUDataset(root='./datasets/MUTAG', name='MUTAG')
    elif dataset_name == 'PROTEINS':
        dataset = TUDataset(root='./datasets/PROTEINS', name='PROTEINS')
    elif dataset_name == 'REDDIT-BINARY':
        dataset = TUDataset(root='./datasets/REDDIT-BINARY', name='REDDIT-BINARY')
    elif dataset_name == 'COX2':
        dataset = TUDataset(root='./datasets/COX2', name='COX2')
    elif dataset_name == 'NCI1':
        dataset = TUDataset(root='./datasets/NCI1', name='NCI1')
    elif dataset_name == 'DD':
        dataset = TUDataset(root='./datasets/DD', name='DD')
    elif dataset_name == 'COLLAB':
        dataset = TUDataset(root='./datasets/COLLAB', name='COLLAB')
    elif dataset_name == 'IMDB-BINARY':
        dataset = TUDataset(root='./datasets/IMDB-BINARY', name='IMDB-BINARY')
    elif dataset_name == 'PPI':
        dataset = PPI(root='./datasets/PPI')
    elif dataset_name == 'QM9':
        dataset = QM9(root='./datasets/QM9')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    
    dataset_with_id = []
    maxd = torch.tensor(100)
    if dataset[0].x is None:
        # 如果节点特征不存在，使用get_dataset_one方法生成全1特征
        for i in range(len(dataset)):
            data = dataset[i]
            data.idx = i
            row, _ = data.edge_index
            if data.x is None:
                num = data.num_nodes
            else:
                num = data.x.shape[0]
            deg = degree(row, num).view((-1, 1))
            deg_capped = torch.min(deg, maxd).type(torch.int64)
            deg_onehot = F.one_hot(deg_capped.view(-1), num_classes=int(maxd.item()) + 1).type(deg.dtype)
            data.x = deg_onehot
            dataset_with_id.append(data)
        return dataset_with_id
    else:
        return dataset
    
#加载人工数据集
def load_syn(dataset_name):
    dataset = Synthetic_Dataset(root='data/synthetic_data')
    return dataset

#如果想得到人工构造数据集
#dataset = MyDataset(create_dataset(100, (10, 20), (0.05, 0.2)))

#下面是4中增强得到子图的方法
def drop_nodes(data):
    '''随机丢弃10%的节点'''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    return data

def permute_edges(data):
    '''随机删除10%的边'''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def subgraph(data):
    '''随机选择20%的节点，构成子图'''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if n not in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in range(len(idx_nondrop))}

    adj = torch.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    # 创建data的副本并进行修改
    new_data = deepcopy(data)
    new_data.edge_index = edge_index

    # 将节点索引转换为相对于子图的索引
    new_data.edge_index[0] = torch.tensor([idx_dict.get(n.item(), -1) for n in new_data.edge_index[0]])
    new_data.edge_index[1] = torch.tensor([idx_dict.get(n.item(), -1) for n in new_data.edge_index[1]])

    # 移除无效的边索引
    mask = new_data.edge_index[0] != -1
    new_data.edge_index = new_data.edge_index[:, mask]

    # 更新节点特征和标签
    new_data.x = new_data.x[idx_nondrop]#有的数据集就没有x
    #bew_num_nodes = len(idx_nondrop)
    #new_data.x = torch.ones(new_num_nodes, 1)
    new_data.y = new_data.y

    return new_data

def mask_nodes_fea(data):
    '''随机选择10%的节点，将其特征随机化'''
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data

def compute_matcher(target_nx, neg_target_nx, result_queue):
    matcher = nx.algorithms.isomorphism.GraphMatcher(target_nx, neg_target_nx)
    result_queue.put(matcher)

class GraphMatchDataset(object):
    def __init__(self, dataset_name, aug = True):
        self.dataset_name = dataset_name
        if dataset_name == 'Synthetic':
            self.dataset = load_syn(self.dataset_name)
        else:
            self.dataset = load_dataset(self.dataset_name)
        self.aug = aug

    def _get_graph(self, idx=None):
        '从数据集中选择一个图Get the idx-th graph in the dataset'
        #idx随机选择
        if idx is None:
            idx = torch.randint(len(self.dataset), (1,)).item()
        return self.dataset[idx]

    def _get_pair(self):
        '使用四种增强方法的某一种对图进行增强，得到一对图,调用_get_graph()方法'
        g = self._get_graph()
        if self.aug == True:
            # aug_g = subgraph(g)#这里暂时使用子图方法进行采样,还可以是drop_nodes,permute_edges,mask_nodes_fea
            # aug_g = permute_edges(g)#删除边
            aug_g = drop_nodes(g)
        else:
            aug_g = g
        return g, aug_g

    def pairs(self, batch_size):#'生成一对图和一个标签'
        # while True:
        batch_graphs = []
        batch_labels = []
        for _ in tqdm(range(batch_size), desc='正在采样正样本'):
            g1, g2 = self._get_pair()
            batch_graphs.append((g1, g2))
            batch_labels.append(1)
        # packed_graphs = self._pack_batch(batch_graphs)
        packed_graphs = batch_graphs#我已经使用了pyg.data类型了，不需要再用这个方法了
        labels = np.array(batch_labels, dtype=np.int32)
        return packed_graphs, labels#我之前使用的是yield
    
    def neg_pairs(self, batch_size):#仅仅判断是不是同构，容易卡住
        #生成一对不匹配图和一个标签
        batch_neg_graphs = []
        batch_neg_labels = []
        for _ in tqdm(range(batch_size), desc='正在采样负样本'):
            target = self._get_graph()
            neg_target = self._get_graph()
            neg_q = subgraph(neg_target)
            #这个判断实在是太慢了，我想办法优化一下，没想到办法优化前先去掉吧
            target_nx = utils.to_networkx(target)
            # neg_q_nx = utils.to_networkx(neg_q)
            neg_target_nx = utils.to_networkx(neg_target)
            matcher = nx.algorithms.isomorphism.GraphMatcher(target_nx, neg_target_nx)#之前是neg_q_nx
            if matcher.subgraph_is_isomorphic():
                continue
            batch_neg_graphs.append((target, neg_target))#之前是neg_q
            batch_neg_labels.append(0)
        return batch_neg_graphs, batch_neg_labels
    
    def neg_pairs0(self, batch_size):#不进行任何判断
        batch_neg_graphs = []
        batch_neg_labels = []
        for _ in tqdm(range(batch_size), desc='正在采样负样本'):
            target = self._get_graph()
            neg_target = self._get_graph()
            batch_neg_graphs.append((target, neg_target))
            batch_neg_labels.append(0)
        return batch_neg_graphs, batch_neg_labels

    def neg_pairs1(self, batch_size):#只能用在linux上
        '生成一对不匹配图和一个标签'
        while True:
            batch_neg_graphs = []
            batch_neg_labels = []
            for _ in tqdm(range(batch_size), desc='正在采样负样本'):#这个方法在整个数据集上循环了，pair是在batch_size上循环
                #随机选取train_dataset中除了target的一个图
                neg_target = random.choice(self.dataset)#好像并没有除去target
                target = random.choice(self.dataset)
                neg_q = subgraph(neg_target)#这里暂时使用子图方法进行采样,还可以是drop_nodes,permute_edges,mask_nodes_fea
                #将target和neg_q转化为networkx的图
                target_nx = utils.to_networkx(target)
                neg_q_nx = utils.to_networkx(neg_q)
                #判断matcher，target和neg_q是否子图同构isomorphism
                try:
                    signal.alarm(10)#设置超时时间为10s
                    matcher = nx.algorithms.isomorphism.GraphMatcher(target_nx, neg_q_nx)
                    if matcher.subgraph_is_isomorphic():
                        # print('同构跳过')
                        continue
                except TimeoutError:
                    print('时间太长')
                    continue
                finally:
                    signal.alarm(0)
                # neg_targets.append(target)#还是原来的正样本
                # neg_queries.append(neg_q)#随机采样一个负样本
                batch_neg_graphs.append((target, neg_q))
                batch_neg_labels.append(0)
        return batch_neg_graphs, batch_neg_labels


    def neg_pairs2(self, batch_size, timeout=20):#使用子进程,就是时间有点长
        batch_neg_graphs = []
        batch_neg_labels = []

        for _ in tqdm(range(batch_size), desc='正在采样负样本'):
            target = self._get_graph()
            neg_target = self._get_graph()
            target_nx = utils.to_networkx(target)
            neg_target_nx = utils.to_networkx(neg_target)

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=compute_matcher, args=(target_nx, neg_target_nx, result_queue))
            process.start()

            try:
                process.join(timeout)

                if process.is_alive():
                    process.terminate()
                    process.join()
                    print('\nlong time, escape!!!')
                    continue  # 跳过当前循环，进入下一个循环

                else:
                    matcher = result_queue.get()
                    if matcher is not None and not matcher.subgraph_is_isomorphic():
                        batch_neg_graphs.append((target, neg_target))
                        batch_neg_labels.append(0)
                    else:
                        print('\nthe luck subgraph')
                        continue  # 跳过当前循环，进入下一个循环

            except KeyboardInterrupt:
                process.terminate()
                process.join()
                raise

        return batch_neg_graphs, batch_neg_labels

    def pack_pair(self, pack_size):
        '将pairs和neg_pairs的返回值合并'
        pos_graphs,pos_labels = self.pairs(pack_size)
        neg_graphs,neg_labels = self.neg_pairs0(pack_size)#正常应该用Neg_pairs,如果对于复杂图应该使用neg_pairs0,因为
        graphs = pos_graphs + neg_graphs
        # labels = np.concatenate((pos_labels + neg_labels))
        labels = [1]*len(pos_graphs) + [0]*len(neg_graphs)
        return graphs, labels
        #return torch.tensor(graphs), torch.tensor(labels, dtype=torch.int64)


# class ValGraphMatchDataset(GraphMatchDataset):
#     def __init__(self, dataset_name, seed=123):
#         super(ValGraphMatchDataset, self).__init__(dataset_name)
#         self.seed = seed
    
#     def pairs(self, idx):
#         '生成一对图和一个标签'
#         pass
    
#人工数据集，目前用处并不大
def create_data(n_nodes_range, p_edge_range):
    n_min, n_max = n_nodes_range
    p_min, p_max = p_edge_range
    n_nodes = torch.randint(n_min, n_max+1, (1,))  # 生成一个节点数
    #p_edge = torch.rand(1).item() * (p_max - p_min) + p_min  # 生成一个边概率
    p_edge = np.random.uniform(p_min, p_max)#随机生成一个边概率
    n_trials = 100

    for _ in range(n_trials):
        g = nx.erdos_renyi_graph(n_nodes.item(), p_edge)
        if nx.is_connected(g):
            # 转换为 PyTorch Geometric 的 Data 类型
            edge_index = torch.tensor(list(g.edges)).t().contiguous()
            x = torch.tensor([[1]] * n_nodes.item(), dtype=torch.float)  # 节点特征
            y = torch.tensor([0], dtype=torch.long)  # 图的标签, 目前就只有0，没有什么含义

            data = Data(x=x, edge_index=edge_index, y=y)
            return data

    raise RuntimeError('Failed to generate a connected graph in {} trials.'.format(n_trials))

def create_dataset(n_samples, n_nodes_range, p_edge_range):
    dataset = []
    for _ in range(n_samples):
        data = create_data(n_nodes_range, p_edge_range)
        dataset.append(data)
    #pyg_dataset = from_data_list(dataset)#pyg中没有from_data_list方法
    #return pyg_dataset
    return dataset#返回的是一个列表