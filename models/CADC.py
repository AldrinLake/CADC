import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, CosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import time
import copy
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

"""
CADC: Class-Aware Drift Compensation for Non-Uniform Semantic Shift in Continual Learning
"""
T = 2
EPSILON = 1e-8

class CADC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["cosine"]:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = CosineIncrementalNet(args, True)
            else:
                self._network = CosineIncrementalNet(args, False)
        else:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = IncrementalNet(args, True)
            else:
                self._network = IncrementalNet(args, False)

        self._protos = []
        self._covs = []
        self._radiuses = []
        self.granularball_list_obj = None

        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_milestones =args['init_milestones']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.milestones = args['milestones']
        self.lrate_decay = args['lrate_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']
        self.w_kd = args['w_kd']
        self.w_trsf = args['w_trsf']
        self.drift_scale = args['drift_scale']
        self.drift_weight = args.get('drift_weight', True)
        self.drift_dist = args.get('drift_dist', "euclidean") # euclidean / cosine
        self.use_past_model = args['use_past_model']
        self.save_model = args['save_model']
        self.model_dir = args['model_dir']
        self.dataset = args['dataset']
        self.init_cls = args['init_cls']
        self.increment = args['increment']
        self._process_id = args['process_id']

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if self.save_model:     
            path = self.model_dir + "{}/{}".format(self.dataset, self.args['seed'])
            if not os.path.exists(path):
                os.makedirs(path)
            self.save_checkpoint("{}/{}_{}".format(path, self.init_cls,self.increment))


    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        self.shot = None
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train",)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # GPU
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.to(self._device)
        model_dir = "{}/{}/{}/{}_{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"], self.args['seed'], self.args["init_cls"],self.args["increment"],self._cur_task)
        if self._cur_task == 0:
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._network.init_de()
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = optim.SGD(self._network.parameters(), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._update_memory(train_loader)
        self._build_protos()
        
    def _drift_estimator(self, train_loader):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network

        optimizer = optim.Adam(_network.drift_estimator.parameters(), lr=0.001)
        for epoch in range(20):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                feats_old = self._old_network(inputs)["features"] 
                feats_new = _network(inputs)["features"]
                x_proj = _network.drift_estimator(feats_old)['logits']
                
                loss = torch.nn.MSELoss()(x_proj, feats_new)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    
    def _update_memory(self, train_loader):
        self._drift_estimator(train_loader)
        self._network.drift_estimator.eval()
        # compute weight
        proto_features_raw = torch.tensor(np.array(self._protos)).to(self._device).to(torch.float32)
        trsf_proto = self._network.drift_estimator(proto_features_raw)['logits'].detach().clone()
        if self.drift_dist == "euclidean":
            drift_strength = torch.norm(trsf_proto - proto_features_raw, p=2, dim=1)  # [N]
        elif self.drift_dist == "cosine":
            drift_strength = 1 - F.cosine_similarity(trsf_proto, proto_features_raw, dim=1)
        else:
            drift_strength = torch.norm(trsf_proto - proto_features_raw, p=2, dim=1)  # [N]

        if self.drift_weight is True:
            drift_weight = drift_strength / (drift_strength.max() + 1e-8)  # [N]
            drift_weight = self.drift_scale * torch.sigmoid(drift_weight)
        else:
            drift_weight = torch.ones_like(drift_strength)
        # ************************
        logging.info(str(drift_weight.cpu().tolist()))
        with torch.no_grad():
            for cls_index in range(0, self._known_classes):
                tmp = self._network.drift_estimator(torch.tensor(self._protos[cls_index]).to(self._device).to(torch.float32))['logits'].detach()
                self._protos[cls_index] = self._protos[cls_index] + drift_weight[cls_index].cpu().numpy() * (tmp.cpu().numpy() - self._protos[cls_index])
        self._network.drift_estimator.train()               

    def _build_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',mode='test', shot=self.shot, ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0) # vectors.mean(0)
            self._protos.append(class_mean)
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_cont = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                features = outputs["features"]
                logits = self._network.fc(features)['logits']
                
                # loss1: new sample classification
                loss_new_cls = F.cross_entropy(logits, targets) 
                L_new_cls += loss_new_cls.item()

                loss =  loss_new_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_cont / len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_cont / len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network

        prog_bar = tqdm(range(self.epochs), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        drift_weight = torch.ones(len(self._protos)).to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_kd = 0.0
            L_trsf = 0.0
            correct, total = 0, 0
            
            
            proto_features_raw = torch.tensor(np.array(self._protos)).to(self._device).to(torch.float32)
            trsf_proto = _network.drift_estimator(proto_features_raw)['logits'].detach().clone()
            if self.drift_dist == "euclidean":
                drift_strength = torch.norm(trsf_proto - proto_features_raw, p=2, dim=1)  # [N]
            elif self.drift_dist == "cosine":
                drift_strength = 1 - F.cosine_similarity(trsf_proto, proto_features_raw, dim=1)
            else:
                drift_strength = torch.norm(trsf_proto - proto_features_raw, p=2, dim=1)  # [N]
            
            drift_weight_tmp = drift_strength / (drift_strength.max() + 1e-8)  # [N]
            drift_weight_tmp = self.drift_scale * torch.sigmoid(drift_weight_tmp)

            drift_weight = drift_weight * 1/2 + drift_weight_tmp * 1/2

            for i, (_, inputs, targets) in enumerate(train_loader):
                loss_clf, loss_kd, loss_transfer = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs= _network(inputs)
                features = outputs["features"]
                logits = _network.fc(features)["logits"]
                
                outputs_old = self._old_network(inputs)
                fea_old = outputs_old["features"]
                logits_old = self._old_network.fc(fea_old)["logits"]

                # ---------------- loss1: new sample classification ---------------
                fake_targets = targets - self._known_classes 
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                L_new_cls += loss_clf.item()

                # ---------------- loss2: drift estimator -------------------------
                fea_transfer = _network.drift_estimator(fea_old)["logits"]
                loss_transfer = torch.nn.MSELoss()(features, fea_transfer) * self.w_trsf
                L_trsf += loss_transfer.item()

                
                # ---------------- loss3: kd --------------------------------
                if self.args["use_vanilla_kd"]:
                    loss_kd = _KD_loss(logits[:, : self._known_classes], logits_old, T) * self.w_kd
                else:
                    loss_kd = _weight_KD_loss(logits[:, : self._known_classes], logits_old, T, drift_weight) * self.w_kd

                L_kd += loss_kd.item()

                loss =  loss_clf + loss_kd + loss_transfer

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_kd {:.3f}, L_trsf {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_kd  / len(train_loader), 
                    L_trsf  / len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_kd {:.3f}, L_trsf {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_kd  / len(train_loader), 
                    L_trsf  / len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

def _weight_KD_loss(pred, soft, T, class_weights=None):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    
    if class_weights is not None:
        # class_weights
        assert class_weights.shape[0] == pred.shape[1], "class weight dimensions must match the number of classes"
        # soft
        class_weights = class_weights.view(1, -1).expand_as(soft)
        
        return -1 * torch.mul(soft * class_weights, pred).sum() / pred.shape[0]
    else:
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]