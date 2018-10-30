import torch
# import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import test_utils as tu
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.metrics import f1_score, roc_auc_score
from test_utils import test_sign_pred
import time
import copy

class psne(nn.Module):

    def __init__(self, n_dim, n_node):
        super(psne, self).__init__()
        self.n_dim = n_dim
        self.n_node = n_node
        self.embedding_u = nn.Embedding(self.n_node, n_dim)
        self.embedding_v = nn.Embedding(self.n_node, n_dim)
        self.init_embedding()

        self.linear1 = nn.Sequential(
            nn.Linear(n_dim, 20),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(20),
            # nn.Tanh()
            nn.ReLU(inplace=True)
            )

        self.linear2 = nn.Sequential(
            nn.Linear(20, 30),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(20),
            # nn.Tanh()
            nn.ReLU(inplace=True)
            )
        self.linear3 = nn.Sequential(
            nn.Linear(30, 20),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(20),
            #nn.Tanh()
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
            )
        # self.linear3 = nn.Linear(12, 10)
        # self.linear4 = nn.Linear(10, 10)

        self.task1_1 = nn.Sequential(
            nn.Linear(20, 5),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(10),
            nn.ReLU(inplace=True)
            # nn.Tanh()#
            )
        # self.task1_2 = nn.Sequential(
        #     nn.Linear(10, 10),
        #     #nn.Dropout(0.5),
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(inplace=True)
        #     # nn.Tanh()
        #     )
        self.pmi = nn.Linear(5, 1)

        self.task2_1 = nn.Sequential(
            nn.Linear(20, 5),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(10),
            nn.ReLU(inplace=True)
            # nn.Tanh()
            )
        # self.task2_2 = nn.Sequential(
        #     nn.Linear(10, 10),
        #     #nn.Dropout(0.5),
        #     nn.BatchNorm1d(10),
        #     nn.LeakyReLU(inplace=True)
        #     # nn.Tanh()
        #     )
        self.sign_prob = nn.Linear(5, 1)
        self.sm = nn.Sigmoid()

    def forward(self, s, t):
        emb_u = self.embedding_u(s)
        emb_v = self.embedding_v(t)

        edge = self.linear1(torch.mul(emb_u, emb_v))
        # edge = self.linear2(edge)
        # edge = self.linear3(edge)
        # edge = F.relu(edge)
        # edge = self.linear4(edge)
        # edge = F.relu(edge)

        out1 = self.task1_1(edge)
        #out1 = self.task1_2(out1)
        out1 = self.pmi(out1)

        out2 = self.task2_1(edge)
        #out2 = self.task2_2(out2)
        out2 = self.sign_prob(out2)
        out2 = self.sm(out2)

        return out1, out2

    def init_embedding(self):
        initrange = 0.001
        # initrange = 0.5 / self.n_dim
        self.embedding_u.weight.data.uniform_(-initrange, initrange)
        self.embedding_v.weight.data.uniform_(-initrange, initrange)

    def save_embedding(self, filenamepre):
        embedding_u = self.embedding_u.weight.data.numpy()
        np.savetxt(filenamepre + '.B.txt', embedding_u, delimiter=',')
        embedding_v = self.embedding_v.weight.data.numpy()
        np.savetxt(filenamepre + '.W.txt', embedding_v, delimiter=',')



def main(data = 'wke', des = 'cv', des2 = '0', n_dim = 20, beta = 0.01, lr = 1e-1, \
        lam = 5e-5, use_gpu = False, seed=1, max_itr = 1000, train = 1, resultdir = 'result_nsne/'):
    print('nSNE: k:%d, data:%s, des:%s, des2:%s,beta:%.4f,lr:%f,lam:%.6f' % (n_dim, data, des, des2, beta,lr,lam))

    torch.manual_seed(seed)

    if use_gpu and torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    n = -1
    if data == 'ep1':
        n = 27503
    elif data == 'wke':
        n = 21535
    elif data == 'wkr':
        n = 11259
    elif data == 'sl1':
        n = 77350
    elif data == 'wke1':
        n = 20198
    
    if des == 'cv':# des2=0~4
        train_edge_file = '%s.cv%s.train.txt' % (data, des2)
        test_edge_file = '%s.cv%s.test.txt' % (data, des2)
    elif des == 't':# des2=0.2,0.4,0.6,0.8
        train_edge_file = '%s.train_%s.txt' % (data, des2)
        test_edge_file = '%s.test_%s.txt' % (data, des2)
    elif des == 'n':# des2 = 0.1;1.0;5.0;10.0
        train_edge_file = '%s.noisy.train.%s.txt' % (data, des2)
        test_edge_file = '%s.noisy.test.%s.txt' % (data, des2)
    elif des == 'all':
        train_edge_file = '%s.edges' % (data)
        # test_edge_file = 'wke.cv5.test.0.txt'

    resultfile = open(resultdir+'n.%s.%s.%s.%d.b%.4f,r%.2f,l%.6f.report.txt'  % (data, des, des2,n_dim, beta,lr,lam), 'w')

    train_edges = np.loadtxt(train_edge_file, dtype=np.int32) 
    
    train_edges[:, 2] = (train_edges[:, 2] + 1) / 2

    train_edges2 = train_edges.copy()
    train_edges2[:,[0,1]] = train_edges2[:,[1,0]]
    train_edges = np.concatenate((train_edges, train_edges2), axis = 0)

    if des != 'all':
        test_edges = np.loadtxt(test_edge_file, dtype=np.int32)
        test_edges[:, 2] = (test_edges[:, 2] + 1) / 2
        n_val = int(test_edges.shape[0]/2)
    if train:
        n_train = len(train_edges)

        deg_matrix = dok_matrix((n, n))
        for line in train_edges:
            deg_matrix[line[0], line[1]] = 1

        deg_all = deg_matrix.nnz
        deg_out = deg_matrix.sum(0).getA1().astype(np.int)
        deg_in = deg_matrix.sum(1).getA1().astype(np.int)
        deg_out[deg_out == 0] = 1
        deg_in[deg_in == 0] = 1

        out_list = deg_out[train_edges[:, 0]]
        in_list = deg_in[train_edges[:, 1]]

        entry = np.ones(n_train) * deg_all / 20
        entry = entry / out_list / in_list
        pmi_list = np.log(entry)

        # build model
        model = psne(n_dim, n)
        if use_gpu:
            model.cuda()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_()

        criterion1 = nn.MSELoss()
        criterion2 = nn.BCELoss()


        # optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=lam, momentum=0.1)
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lam)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=lam)

        if use_gpu:
            s_list = Variable(torch.LongTensor(train_edges[:, 0].tolist()).cuda())
            t_list = Variable(torch.LongTensor(train_edges[:, 1].tolist()).cuda())
            sign_list = Variable(torch.Tensor(train_edges[:, 2].tolist()).cuda())
            pmi_list = Variable(torch.Tensor(pmi_list).cuda())
            if des != 'all':
                test_s_list = Variable(torch.LongTensor(test_edges[:, 0].tolist()).cuda())
                test_t_list = Variable(torch.LongTensor(test_edges[:, 1].tolist()).cuda())
        else:
            s_list = Variable(torch.LongTensor(train_edges[:, 0].tolist()))
            t_list = Variable(torch.LongTensor(train_edges[:, 1].tolist()))
            sign_list = Variable(torch.Tensor(train_edges[:, 2].tolist()))
            pmi_list = Variable(torch.Tensor(pmi_list))
            if des != 'all':
                test_s_list = Variable(torch.LongTensor(test_edges[:, 0].tolist()))
                test_t_list = Variable(torch.LongTensor(test_edges[:, 1].tolist()))

        lastloss = 0
        result = ''
        total_time = time.time()
        max_f1 = 0
        patience = 100
        for epoch in range(max_itr):

            print('*' * 10)
            print('epoch: {}'.format(epoch + 1))

            epoch_stime = time.time()

            running_loss1 = 0
            running_loss2 = 0

            # forward
            out1, out2 = model(s_list, t_list)
            loss1 = criterion1(out1.squeeze(), pmi_list)
            loss2 = criterion2(out2.squeeze(), sign_list)
            running_loss1 += beta * loss1.item() * 2
            running_loss2 += (1-beta) * loss2.item() * 2

            total_loss = beta * loss1 + (1-beta) * loss2

            # backward
            optimizer.zero_grad()
            # loss1.backward(retain_variables=True)
            # loss2.backward()
            total_loss.backward()

            optimizer.step()
            # if epoch>20 and epoch%10==0:
            #     model.save_embedding(resultfilename+'.t{}'.format(epoch))
            print('Loss: {:.6f} {:.6f} {:.6f} {} Time:{:.2f}'.format(running_loss1,
                                                      running_loss2, total_loss.item(),result, time.time()-epoch_stime))
            if des == 'all':
                if lastloss != 0 and (lastloss < total_loss.item() or abs(lastloss-total_loss.item()) < 0.0001):
                    break
                lastloss = total_loss.item()

            else:
                result, val_f1, val_ruc = test_sign_pred(model, test_s_list[:n_val], test_t_list[:n_val], test_edges[:n_val,2], use_cuda = use_gpu)
                model.train()
                val_f1 = round(val_f1, 3)
                if val_f1 > max_f1:
                    # cache
                    cachemodel = copy.deepcopy(model.state_dict())
                    # print(cachemodel)
                    max_f1 = val_f1
                    cov_time = time.time() - total_time
                    patience = 100
                else:
                    patience -= 1
            if patience == 0:
                # print(model.state_dict())
                # print(cachemodel)
                model.load_state_dict(cachemodel)
                break

        # model.save_embedding(resultfilename)
        # test
        print('Total Time: {:.1f}, Coverage Time: {:.1f}'.format((time.time() - total_time), cov_time))
        model.eval()

        if use_gpu:
            B = model.embedding_u.weight.data.cpu().numpy()
            W = model.embedding_v.weight.data.cpu().numpy()
        else:
            B = model.embedding_u.weight.data.numpy()
            W = model.embedding_v.weight.data.numpy()
        # np.savetxt(resultdir+'n.%s.%s.%s.%d.b%.4f,r%.2f,l%.6f.B.txt'  % (data, des, des2,n_dim, beta,lr,lam),B)
        # np.savetxt(resultdir+'n.%s.%s.%s.%d.b%.4f,r%.2f,l%.6f.W.txt'  % (data, des, des2,n_dim, beta,lr,lam),W)
    else:
        B = np.loadtxt(resultdir+'n.%s.%s.%s.%d.b%.4f,r%.2f,l%.6f.B.txt'  % (data, des, des2,n_dim, beta,lr,lam))
        W = np.loadtxt(resultdir+'n.%s.%s.%s.%d.b%.4f,r%.2f,l%.6f.W.txt'  % (data, des, des2,n_dim, beta,lr,lam))

    if des != 'all':
        if train:
            result, f1, ruc = test_sign_pred(model, test_s_list[n_val:], test_t_list[n_val:], test_edges[n_val:,2], use_cuda = use_gpu)
    
            result = 'f1/ruc: {:.6f} {:.6f}'.format(f1, ruc)
            print(result)
            resultfile.write('sign f: \n%f\t%f\nCoverage Time: %1f' % (f1, ruc, cov_time))

    if data == 'wke1' and des == 'all':

        node_labels = np.loadtxt('wke1.c.txt', dtype = int)

        acc, f1,auc = tu.node_classification(node_labels, [B, W])
        print(acc,f1, auc)
        resultfile.write('LR node classification:\n%f\t%f\t%f\n' % (acc,f1,auc))


import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default = 20)
    parser.add_argument('--data', type=str, default = 'wkr')
    parser.add_argument('--des', type=str, default = 'cv')
    parser.add_argument('--des2', type=str, default = '0')
    parser.add_argument('--beta', type=float, default = 0.05)
    parser.add_argument('--lr', type=float, default = 0.01)
    parser.add_argument('--lam', type=float, default = 1e-6)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--dir', type=str, default = 'result_nsne/')
    args = parser.parse_args()

    main(data = args.data, des = args.des, des2 = args.des2, n_dim = args.k, beta = args.beta, lr = args.lr,
     lam = args.lam, use_gpu = True, seed=1, max_itr = 1000,train = args.train, resultdir = args.dir)