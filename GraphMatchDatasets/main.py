from dataset import *
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np

def main():
    dataset_name = 'ENZYMES'  # 替换为你要使用的数据集名称
    augmentation = True  # 是否进行数据增强

    dataset = GraphMatchDataset(dataset_name, augmentation)

    batch_size = 200  # 批处理大小

    # all_graph_pairs, all_labels = dataset.pack_pair(batch_size)
    all_graph_pairs, all_labels = dataset.neg_pairs0(batch_size)
    print(all_graph_pairs, len(all_graph_pairs), type(all_graph_pairs))
    print(all_labels, len(all_labels), type(all_labels))
    # 输出图对和标签
    for graph_pair, label in zip(all_graph_pairs, all_labels):
        g1, g2 = graph_pair

        print("-----")

        g1_nx = utils.to_networkx(g1)
        g2_nx = utils.to_networkx(g2)
        matcher = nx.algorithms.isomorphism.GraphMatcher(g1_nx, g2_nx)
        if matcher.subgraph_is_isomorphic():
            print("True")

def main1():
    dataset_eval = GraphMatchDataset(dataset_name = 'ENZYMES')
    graphs_eval, labels_eval = dataset_eval.pack_pair(pack_size = 50)#原来是len(dataset)
    eval_dataset = list(zip(graphs_eval, labels_eval))
    random.shuffle(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=10)
    print(eval_dataloader)

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

def main2():#测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gconv = GConv(input_dim=1, hidden_dim=5, num_layers=2).to(device)

    dataset_eval = GraphMatchDataset(dataset_name='REDDIT-BINARY')
    graphs_eval, labels_eval = dataset_eval.pack_pair(pack_size = 50)
    eval_dataset = list(zip(graphs_eval, labels_eval))
    random.shuffle(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=10)
    print(eval_dataloader)
    for data in eval_dataloader:
        (graph_t, graph_g), labels = data
        graph_t.to(device)
        graph_g.to(device)
        labels.to(device)
        _, g1 = gconv(graph_t.x, graph_t.edge_index, graph_t.batch)
        _, g2 = gconv(graph_g.x, graph_g.edge_index, graph_g.batch)
        print(g2)


def main3():
    pass

if __name__ == '__main__':
    main()
    # main1()
    # main2()