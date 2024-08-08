import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import time
import random
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from GraphMatchDatasets.dataset import GraphMatchDataset


@torch.no_grad()
def test_save_model(gconv, dataloader, device, path, epoch,
                    logger, t, norm, save_embed, eval_model, save_model):
    path_epo = path + '_' + str(t) + '_' + str(epoch)

    if save_model:
        torch.save(gconv.state_dict(), path_epo + '_model.pkl')

    gconv.eval()
    x = []
    y = []
    for data in dataloader:
        # print(data.idx)
        data = data.to(device)
        _, g = gconv(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)

    x = torch.cat(x, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()

    if save_embed:
        with open(path_epo + '.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(path_epo + ' saved.')

    if eval_model:
        return test_SVM(x, y, logger, t, norm)
    else:
        return None

#改进于test_save_model函数，用于评估
@torch.no_grad()
def test_save_model0(gconv, dataset0, device, path, epoch,
                    logger, t, norm, save_embed, eval_model, save_model, args):
    path_epo = path + '_' + str(t) + '_' + str(epoch)
    # 我先不保存模型了
    # if save_model:
    #     torch.save(gconv.state_dict(), path_epo + '_model.pkl')
    
    dataset_test = GraphMatchDataset(dataset_name=args.dataset)
    graphs, labels = dataset_test.pack_pair(pack_size=int(len(dataset0)/3))#我感觉这里应该改成数据集的1/3
    test_dataset = list(zip(graphs, labels))
    random.shuffle(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    gconv.eval()
    x1, x2, y =[], [], []
    for data in test_dataloader:
        (graph_t, graph_g), labels = data
        graph_t.to(device)
        graph_g.to(device)
        labels.to(device)
        _, g1 = gconv(graph_t.x, graph_t.edge_index, graph_t.batch)
        _, g2 = gconv(graph_g.x, graph_g.edge_index, graph_g.batch)
        x1.append(g1)
        x2.append(g2)
        y.append(labels)

    x1 = torch.cat(x1, dim=0).cpu().detach().numpy()
    x2 = torch.cat(x2, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()
    x = np.concatenate((x1, x2), axis=1)

    # if save_embed:
    #     with open(path_epo + '.pkl', 'wb') as f:
    #     # with open(path + path_epo + '.pkl', 'wb') as f:
    #         pickle.dump({'x': x, 'y': y}, f)
    #     logger.info(path_epo + ' saved.')

    if eval_model:
        return test_SVM(x, y, logger, t, norm)
    else:
        return None


@torch.no_grad()
def test_save_model2(gconv1, gconv2, dataloader, device, path, epoch,
                     logger, t, norm, save_embed, eval_model, save_model):
    path_epo1 = path + '_' + str(t) + '_' + str(epoch) + '_1'
    path_epo2 = path + '_' + str(t) + '_' + str(epoch) + '_2'
    path_epo = path + '_' + str(t) + '_' + str(epoch)

    if save_model:
        torch.save(gconv1.state_dict(), path_epo1 + '_model.pkl')
        torch.save(gconv2.state_dict(), path_epo2 + '_model.pkl')

    gconv1.eval()
    gconv2.eval()
    x1, x2, y = [], [], []
    for data in dataloader:
        data = data.to(device)
        _, g1 = gconv1(data.x, data.edge_index, data.batch)
        _, g2 = gconv2(data.x, data.edge_index, data.batch)
        x1.append(g1)
        x2.append(g2)
        y.append(data.y)

    x1 = torch.cat(x1, dim=0).cpu().detach().numpy()
    x2 = torch.cat(x2, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()
    x = np.concatenate((x1, x2), axis=1)

    if save_embed:
        with open(path_epo + '.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(path_epo + ' saved.')

    if eval_model:
        return test_SVM(x, y, logger, t, norm)
    else:
        return None

#用于评估k-core、k-truss增强后子图匹配情况，改进于test_save_model2
@torch.no_grad()
def test_save_model3(gconv1, gconv2, dataloader, device, path, epoch,
                     logger, t, norm, save_embed, eval_model, save_model, args):
    path_epo1 = path + '_' + str(t) + '_' + str(epoch) + '_1'
    path_epo2 = path + '_' + str(t) + '_' + str(epoch) + '_2'
    path_epo = path + '_' + str(t) + '_' + str(epoch)

    if save_model:
        torch.save(gconv1.state_dict(), path_epo1 + '_model.pkl')
        torch.save(gconv2.state_dict(), path_epo2 + '_model.pkl')
    dataset_test = GraphMatchDataset(dataset_name=args.dataset)
    graphs, labels = dataset_test.pack_pair(pack_size= args.batch_size)#int(len(dataloader)/3)
    test_dataset = list(zip(graphs, labels))
    test_dataset1 = random.shuffle(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size)

    gconv1.eval()
    gconv2.eval()
    x1, x2, y =[], [], []
    for data in test_dataloader:
        (graph_t, graph_g), labels = data
        #data = data.to(device)
        graph_t.to(device)
        graph_g.to(device)
        labels.to(device)
        _, g1 = gconv1(graph_t.x, graph_t.edge_index, graph_t.batch)
        _, g2 = gconv2(graph_g.x, graph_g.edge_index, graph_g.batch)
        x1.append(g1)
        x2.append(g2)
        y.append(labels)

    x1 = torch.cat(x1, dim=0).cpu().detach().numpy()
    x2 = torch.cat(x2, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()
    x = np.concatenate((x1, x2), axis=1)

    if save_embed:
        with open(path_epo + '.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(path_epo + ' saved.')

    if eval_model:
        return test_SVM(x, y, logger, t, norm)
    else:
        return None

def test_LR(x, y, logger, t, norm):
    if norm:
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)

    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(x_train, y_train)
        accs.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
    logger.info("Acc:{:.2f}±{:.2f}".format(np.mean(accs), np.std(accs)))
    return np.mean(accs), np.std(accs)


def test_SVM0(x, y, logger, t, norm=False, search=True):#没有计算AUROC的版本
    y = LabelEncoder().fit_transform(y)
    if norm:
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)

    logger.info("accuracies_val:{:.2f}±{:.2f}, accuracies:{:.2f}±{:.2f}".format(
        np.mean(accuracies_val), np.std(accuracies_val), np.mean(accuracies), np.std(accuracies)))
    return np.mean(accuracies_val), np.mean(accuracies)

def test_SVM(x, y, logger, t, norm=False, search=True):
    y = LabelEncoder().fit_transform(y)
    if norm:#如果设置了归一化，则使用MinMaxScaler对特征进行归一化处理
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)#使用10折交叉验证
    accuracies = []
    accuracies_val = []
    aurocs = []
    aurocs_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            # classifier = GridSearchCV(SVC(), params, cv=5, scoring=['accuracy', 'roc_auc'], verbose=0)
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        aurocs.append(roc_auc_score(y_test, classifier.decision_function(x_test)) * 100)

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        aurocs_val.append(roc_auc_score(y_test, classifier.decision_function(x_test)) * 100)

    logger.info("accuracies_val:{:.2f}±{:.2f}, accuracies:{:.2f}±{:.2f}".format(
        np.mean(accuracies_val), np.std(accuracies_val), np.mean(accuracies), np.std(accuracies)))
    logger.info("aurocs_val:{:.2f}±{:.2f}, aurocs:{:.2f}±{:.2f}".format(
        np.mean(aurocs_val), np.std(aurocs_val), np.mean(aurocs), np.std(aurocs)))
    return np.mean(accuracies_val), np.mean(accuracies), np.mean(aurocs_val), np.mean(aurocs)


# def evaluate(mean, std, logger, interval):
#     logger.info("### Results ###")
#     for i in range(len(mean)):
#         logger.info("Epoch: {:}, Accuracy: {:.2f}±{:.2f}".format((i + 1) * interval, mean[i], std[i]))
#     logger.info("### Best Accuracy ###")
#     index = np.argmax(mean)
#     logger.info("Epoch: {:}, Best Accuracy: {:.2f}±{:.2f}".format((index + 1) * interval, mean[index], std[index]))
#
#
# def final_eval(accs, logger, interval):
#     mean = np.mean(accs, axis=0)
#     std = np.std(accs, axis=0)
#     logger.info("### Results ###")
#     for i in range(len(mean)):
#         logger.info("Epoch: {:}, Accuracy: {:.2f}±{:.2f}".format((i + 1) * interval, mean[i], std[i]))
#     logger.info("### Best Accuracy ###")
#     index = np.argmax(mean)
#     logger.info("Epoch: {:} , Best Accuracy: {:.2f}±{:.2f}".format((index + 1) * interval, mean[index], std[index]))
