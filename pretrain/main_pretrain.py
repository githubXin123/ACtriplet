import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime
from queue import Queue
from threading import Thread
import pandas as pd
import math


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

from model_pretrain import GIN_AC, Prot_trans


import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

def triplet_loss(model_anchor, model_positive, model_negative, margin):
    distance1 = torch.sqrt(torch.sum(torch.pow(model_anchor - model_positive, 2), dim=1, keepdim=True))
    distance2 = torch.sqrt(torch.sum(torch.pow(model_anchor - model_negative, 2), dim=1, keepdim=True))
    return torch.mean(torch.relu(distance1 - distance2 + margin))


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_pretrain.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class AC_pretrain(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        #./ckpt/dir_name
        self.log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.dataset = dataset
        self.device = self._get_device()

        # load the protein embedding
        self.protein_path = os.path.join('../data_pre/protein_emb', config['protein_file'])
        self.protein_feat = torch.tensor(np.loadtxt(self.protein_path, delimiter=',')).float().to(self.device)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    # AC triplet loss
    def _step1(self, model, data):
        # get the embedding of the three molecules
        anchor_mol = model(data[0], self.protein_feat)
        AC_mol = model(data[1], self.protein_feat)
        nonAC_mol = model(data[2], self.protein_feat)
        
        # normalize the embedding
        anchor_mol = F.normalize(anchor_mol, dim=1)
        AC_mol = F.normalize(AC_mol, dim=1)
        nonAC_mol = F.normalize(nonAC_mol, dim=1)

        # compute the loss
        margin_tem = abs(data[1].AC_pki - data[2].nonAC_pki)
        margin_cliff = []
        for i in margin_tem:
            if i >= 0 and i <= self.config['margin_AC']:
                margin_cliff.append(i)
            else:
                margin_cliff.append(self.config['margin_AC'])
        loss_cliff = triplet_loss(anchor_mol, nonAC_mol, AC_mol, torch.tensor(margin_cliff).to(anchor_mol.device))

        return loss_cliff, anchor_mol, nonAC_mol
    
    # bio triplet loss
    def _step2(self, model, data):
        high_mol, protein_emb = model(data[0], self.protein_feat, with_prot = True)
        low_mol = model(data[1], self.protein_feat)

        high_mol = F.normalize(high_mol, dim=1)
        low_mol = F.normalize(low_mol, dim=1)
        protein_emb = F.normalize(protein_emb, dim=1)

        margin_tem = data[0].high_pki - data[1].low_pki
        margin_bio = []
        for i in margin_tem:
            if i >= 0 and i <= self.config['margin_bio']:
                margin_bio.append(i)
            else:
                margin_bio.append(self.config['margin_bio'])

        loss_activity = triplet_loss(protein_emb, high_mol, low_mol, torch.tensor(margin_bio).to(high_mol.device))
        return loss_activity

    def train_step(self, bio_dataset):
        train_loader, valid_loader, AC_num = self.dataset.get_data_loaders()
        bio_train_loader = bio_dataset.get_data_loaders()
        print('number of AC molecules in the dataset：', AC_num)

        model = GIN_AC(**self.config["model"]).to(self.device)
        model = self.load_pre_trained_weights(model, self.config['pretrain_path'], self.config['num_model'])
        print(model)


        layer_list = []
        for name, param in model.named_parameters():
            if 'trans_prot' in name:
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer1 = torch.optim.Adam(
            [{'params': params, 'lr': 0}, {'params': base_params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )
        optimizer2 = torch.optim.Adam(
            [{'params': params, 'lr': self.config['bio_lr']}, {'params': base_params, 'lr': self.config['biognn_lr']}],
            weight_decay=eval(self.config['weight_decay'])
        )
        
        warmup_epochs = self.config['warmup']
        scheduler = CosineAnnealingLR(optimizer1, T_max=self.config['T_max'], eta_min=self.config['lr_min'])
        scheduler_warmup = LambdaLR(optimizer1, lr_lambda=lambda epoch: epoch/warmup_epochs)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=self.config['epochs'], eta_min=self.config['lr_min'])

        # path to save the checkpoints
        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')

        # save the config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        n_iter_bio = 0

        best_valid_loss = np.inf
        epochs_without_improvement = 0
        max_epochs_without_improvement = self.config['max_epochs_without_improvement']
        
        valid_loss_list = []

        AC_batch_num = math.ceil(AC_num / self.config['batch_size'])
        for epoch_counter in range(self.config['epochs']):
            if epoch_counter < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler.step()
            print(" for the %d epoch, the learning rate is ：%f" % (epoch_counter, optimizer1.param_groups[1]['lr']))


            self.writer.add_scalar('AC_lr_gnn', optimizer1.param_groups[1]['lr'], global_step=epoch_counter)
            self.writer.add_scalar('AC_lr_prot', optimizer1.param_groups[0]['lr'], global_step=epoch_counter)


            for i in range(self.config['num_biopretrain']):
                for bn, a in enumerate(bio_train_loader):
                    if bn >= AC_batch_num:
                        break
                    for i in range(2):
                        a[i] = a[i].to(self.device, non_blocking=True)
                
                    optimizer2.zero_grad()
                    loss_bio = self._step2(model, a)

                    if n_iter_bio % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('activity_loss', loss_bio, global_step=n_iter_bio)
                    n_iter_bio += 1
                    loss_bio.backward()
                    optimizer2.step()
            scheduler2.step()

            for i in range(self.config['num_ACpretrain']):
                for bn, a in enumerate(train_loader):
                    for i in range(3):
                        a[i] = a[i].to(self.device, non_blocking=True)
                
                    optimizer1.zero_grad()
                    loss, anchor_emb, nonAC_emb = self._step1(model, a)

                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('AC_loss', loss, global_step=n_iter)



                    loss.backward()
                    optimizer1.step()
                    n_iter += 1

            self.writer.add_scalar('bio_lr_prot', optimizer2.param_groups[0]['lr'], global_step=epoch_counter)
            self.writer.add_scalar('bio_lr_gnn', optimizer2.param_groups[1]['lr'], global_step=epoch_counter)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)

                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch_counter)


    def load_pre_trained_weights(self, model, prefile_path, num_model):
        try:
            checkpoints_folder = os.path.join('./ckpt', prefile_path, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, num_model), map_location=self.device)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        num_data = 0
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for bn, data in enumerate(valid_loader):
                for i in range(3):
                    data[i] = data[i].to(self.device, non_blocking=True)
                loss, _, _ = self._step1(model, data)
                valid_loss += loss.item() * data[0].anchor_pki.size(0)
                num_data += data[0].anchor_pki.size(0)
            valid_loss /= num_data
        model.train()
        print('Valid loss:', valid_loss)
        return valid_loss

def main():
    config = yaml.load(open("./config_pretrain.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)
    print(config)

    from dataset_pretrain import ACDatasetWrapper, ActivityDatasetWrapper
    dataset_AC = ACDatasetWrapper(config['batch_size'], **config['dataset_AC'])
    dataset_bio = ActivityDatasetWrapper(batch_size=config['batch_size'], num_workers=8, data_path=config['bio_path'])

    AC_premodel = AC_pretrain(dataset_AC, config)
    AC_premodel.train_step(dataset_bio)


if __name__ == "__main__":
    main()
