import torch as T
from torch import nn
from torch.nn import functional as F
from .layer import GraphConv, GraphConv_p, GraphPool, GraphOutput, GraphOutput_p,Encoder_1,Decoder_1
import numpy as np
from torch import optim
import time
from .util import dev
from sklearn.metrics import accuracy_score,roc_auc_score


def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)

# https://github.com/vmasrani/gae_in_pytorch/blob/master/models.py
class GraphConvAutoEncoder(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class):
        super(GraphConvAutoEncoder, self).__init__()
        self.encoder = Encoder_1(input_dim=37, conv_width=200)
        self.decoder = Decoder_1(input_dim=200, conv_width=37)
        self.to(dev)

    def forward(self, p_atoms, p_edges):
        summed_features, atoms, neighbor_features, self_features = self.encoder(p_atoms, p_edges)
        out_neighbor_features, out_self_features = self.decoder(neighbor_features, self_features, p_edges)
        return summed_features, atoms, out_neighbor_features, out_self_features

    def fit(self, loader_train, epochs=100, lr=1e-3):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            t0 = time.time()
            for Ab, Bb, Eb, yb in loader_train:
                Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                nf, sf, onf, osf = self.forward(Ab, Eb)
                loss = criterion(nf, onf) + criterion(sf, osf)
                loss.backward()
                optimizer.step()
            print('[Epoch:%d/%d] %.1fs loss_train: %f' % (epoch, epochs, time.time() - t0, loss.item()))
        return

class QSAR(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class):
        super(QSAR, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphConv_p(input_dim=480, conv_width=200)
        self.gcn_pro_2 = GraphConv_p(input_dim=200, conv_width=100)
        self.gop = GraphOutput(input_dim=134, output_dim=hid_dim_m)
        self.gop_p = GraphOutput_p(input_dim=100, output_dim=hid_dim_p)
        self.bn = nn.BatchNorm2d(80)
        self.pool = GraphPool()
        self.fc1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        # self.fc1 = nn.Linear(hid_dim_p , 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, n_class)
        self.to(dev)

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        # for ligand
        m_atoms = self.gcn_mol_1(m_atoms, m_bonds, m_edges)
        m_atoms = F.relu(m_atoms)
        # m_atoms = self.bn(m_atoms)
        # m_atoms = self.pool(m_atoms, m_edges)
        m_atoms = self.gcn_mol_2(m_atoms, m_bonds, m_edges)
        m_atoms = F.relu(m_atoms)
        # m_atoms = self.bn(m_atoms)
        # m_atoms = self.pool(m_atoms, m_edges)
        fp_m = self.gop(m_atoms, m_bonds, m_edges)

        # for pocket
        p_atoms = self.gcn_pro_1(p_atoms, p_edges)
        p_atoms = F.relu(p_atoms)
        # p_atoms = self.bn(p_atoms)
        p_atoms = self.gcn_pro_2(p_atoms, p_edges)
        p_atoms = F.relu(p_atoms)
        # p_atoms = self.bn(p_atoms)
        fp_p = self.gop_p(p_atoms, p_edges)

        fp = T.cat([fp_m, fp_p], dim=1)
        # interaction = T.sigmoid(self.fc1(fp))
        # out = T.softmax(self.fc2(interaction),dim=1)
        tmp = self.fc1(fp)
        
        tmp1 = self.fc3(tmp)
        out = T.sigmoid(self.fc2(tmp1))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=100, lr=1e-5):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            # print(len(loader_train),len(loader_valid))
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                # print("yb:",yb)
                # print('Vb:',Vb)
                # y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                # print("y_:",y_)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                #print('loss:',loss.item()) # batch loss
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                # acc_train += ((y_>0.5).type_as(yb)==yb).float().mean().item()
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)

            # print training info
            ## full info
            # print('[Epoch:%d/%d] %.1fs loss_train: %.3f %.3f loss_valid: %.3f acc_train: %.3f acc_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, time.time() - t0, loss.item(), loss_train, loss_valid, acc_train, acc_valid, auc_train, auc_valid))
            ## simple info
            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, time.time() - t0, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop: break
        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = nn.BCELoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            # acc += ((y_>0.5).type_as(yb)==yb).float().mean().item()
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score