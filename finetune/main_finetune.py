import os
import shutil
import sys
sys.path.append('/home/xxx/SCL_AC')

import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import random


from model_gnn import GIN_AC
from dataset_fintune import ftDatasetWrapper, testDatasetWrapper

import torch
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from MoleculeACE import Data, calc_cliff_rmse

import pickle



class train_and_valid:
    def __init__(self, model, device, mini_epoch, prot_path):

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')

        self.path1 = os.path.join('./temp_model_path', dir_name)
        self.save_path = os.path.join('./temp_model_path', dir_name, 'best_model.pkl')
        self.log_dir = os.path.join('./checkpoints', dir_name)

        self.mini_epoch = mini_epoch

        self.protein_path = os.path.join('../data_pre/protein_emb', prot_path)
        self.protein_feat = torch.tensor(np.loadtxt(self.protein_path, delimiter=',')).float()
        print('the shape of the protein feature:', self.protein_feat.shape)

        self.criterion = nn.MSELoss()
        self.model = model
        self.device = device


    def train_stage(self, train_loader, valid_loader, early_stopping_patience, epochs, test_loader, cliff_mols_test):
        

        os.makedirs(self.path1, exist_ok=True)
        patience = 0
        best_val_loss = np.inf
        best_val_rmse = np.inf

        earlystop_epoch = 0
        n_iter = 0

        for epoch in range(epochs):
            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:
                with open(self.save_path, 'rb') as handle:
                    self.model = pickle.load(handle)
                _, test_pred, test_true = self.predict_stage(test_loader)
                print(f'Stopping training early, epoch {earlystop_epoch}')
                print('path:',self.save_path)

                try:
                    print('valid loss:', best_val_loss, 'valid rmse:', best_val_rmse)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:
                train_loss, n_iter_new = self._one_epoch(train_loader, n_iter)
                n_iter = n_iter_new

                val_loss, valid_pred_temp, valid_true_temp = self.predict_stage(valid_loader)
                _, test_pred_temp, test_true_temp = self.predict_stage(test_loader)
                valid_rmse_temp = mean_squared_error(valid_true_temp, valid_pred_temp, squared=False)
                cliff_rmse,_ ,_ = calc_cliff_rmse(y_test_pred=test_pred_temp, y_test=test_true_temp, cliff_mols_test=cliff_mols_test)

                print('epoch:',epoch,'valid_rmse:', valid_rmse_temp, 'test_rmse:', mean_squared_error(test_true_temp, test_pred_temp, squared=False), 'cliff_rmse:', cliff_rmse)

                if epoch >= self.mini_epoch:

                    if valid_rmse_temp <= best_val_rmse:

                        best_val_rmse = valid_rmse_temp
                        earlystop_epoch = epoch
                        with open(self.save_path, 'wb') as handle:
                            pickle.dump(self.model, handle)
                        patience = 0
                    else:
                        patience += 1
        return epoch, best_val_rmse, test_pred, test_true

    def _one_epoch(self, train_loader, n_iter):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        
        for idx, batch_data in enumerate(train_loader):

            # Move batch to gpu
            batch_data.to(self.device, non_blocking=True)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat, _ = self.model(batch_data, self.protein_feat)

            # Calculating the loss and gradients
            loss = self.loss_fn(y_hat, batch_data.y)

            # self.board_record.add_scalar('train_loss', loss, global_step=n_iter)
            n_iter += 1
            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

        return loss, n_iter

    def predict_stage(self, test_loader):
        y_pred = []
        y_true = []
        valid_loss = 0
        num_data = 0
        with torch.no_grad():
            self.model.eval()
            for batch, data in enumerate(test_loader):
                data.to(self.device, non_blocking=True)
                y_hat, _ = self.model(data, self.protein_feat)
                loss = self.loss_fn(y_hat, data.y)
                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)
                try:
                    y_pred.extend(y_hat.view(-1).squeeze().cpu().detach().numpy())
                except:
                    y_pred.extend(y_hat.view(-1).cpu().detach().numpy())
                y_true.extend(data.y.cpu().flatten().numpy())
        self.model.train()

        return valid_loss/num_data, y_pred, y_true
    
    def loss_fn(self, pred, real_label):
        loss = self.criterion(pred, real_label)
        return loss



    def load_pre_trained_weights(self, prefile_path, num_model, display):
        try:
            checkpoints_folder = os.path.join('../pretrain/ckpt', prefile_path, 'checkpoints')


            state_dict = torch.load(os.path.join(checkpoints_folder, num_model), map_location=self.device)
            self.model.load_my_state_dict(state_dict, display)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        # return
    
    def set_optimizer(self, lr, lr_base, lr_prot, weight_decay):
        
        new_layer_name = ['c_to_p_transform', 'p_to_c_transform', 'mc1', 'mp1', 'hc0', 'hp0', 'hc1', 'hp1', 'GRU_dma', 'W_out','super_final']
        new_layer_list = [name for name, param in self.model.named_parameters() if any(i in name for i in new_layer_name)]
        prot_layer_list = [name for name, param in self.model.named_parameters() if 'trans_prot' in name]
        new_add_prot = new_layer_list + prot_layer_list

        new_layer_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in new_layer_list, self.model.named_parameters()))))
        prot_layer_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in prot_layer_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in new_add_prot, self.model.named_parameters()))))

        self.optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': lr_base}, {'params': prot_layer_params, 'lr': lr_prot}, {'params': new_layer_params}],
            lr, weight_decay=eval(weight_decay)
        )

def set_random_seed(seed=2023):
    print('random seed：', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_pred():
    config_raw = yaml.load(open("./finetune_config.yaml", "r"), Loader=yaml.FullLoader)
    print(config_raw)

    if torch.cuda.is_available() and config_raw['gpu'] != 'cpu':
        device = config_raw['gpu']
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    print("Running on:", device)

    dataset_name = config_raw['dataset_name']
    data_1 = Data(dataset_name)
    
    test_data = testDatasetWrapper(config_raw['batchsize_test'], config_raw['num_workers'], data_1.smiles_test, data_1.y_test, config_raw['prot_path'])
    test_loader = test_data.get_data_loaders()

    # split the data into 5 folds for cross-validation
    ss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cutoff = np.median(data_1.y_train)
    labels = [0 if i < cutoff else 1 for i in data_1.y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    rmse_scores = []
    epoch_record = []
    rmse_cliff = []
    valid_rmse_list = []
    cliff_true_list = []
    cliff_pred_list = []
    count = 1

    
    display = True
    for split in splits:
        set_random_seed(seed=2023)
        init_model = GIN_AC(conv_dim=config_raw['conv_dim'], graph_dim=config_raw['graph_dim'],pred_n_layer=config_raw['pred_n_layer'],
                        pred_act=config_raw['pred_act'], dropout=config_raw['dropout'], pool=config_raw['pool'], protein_dim=config_raw['protein_dim'],
                        num_layer=config_raw['num_conv'], prot_layer=config_raw['prot_trans_layer'], DMA_depth = config_raw['DMA_depth']).to(device)
        mymodel = train_and_valid(init_model, device, config_raw['mini_epoch'], config_raw['prot_path'])
        mymodel.load_pre_trained_weights(config_raw['pretrain_path'], config_raw['num_model'], display)
        mymodel.set_optimizer(config_raw['init_lr'], config_raw['init_base_lr'], config_raw['lr_prot'], config_raw['weight_decay'])

        print(f'*****the{count}fold*****')
        x_tr_fold = [data_1.smiles_train[i] for i in split['train_idx']]
        y_tr_fold = [data_1.y_train[i] for i in split['train_idx']]
        x_val_fold = [data_1.smiles_train[i] for i in split['val_idx']]
        y_val_fold = [data_1.y_train[i] for i in split['val_idx']]

        data_load = ftDatasetWrapper(config_raw['batchsize_train'], config_raw['batchsize_valid'], config_raw['num_workers'], x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, config_raw['prot_path'])

        train_loader, valid_loader = data_load.get_data_loaders()
        if count == 1:
            print(init_model)
            
        epoch, valid_rmse, test_pred, test_true = mymodel.train_stage(train_loader, valid_loader, config_raw['early_stopping_patience'], config_raw['epochs'], test_loader, data_1.cliff_mols_test)
        

        count += 1

        valid_rmse_list.append(valid_rmse)
        epoch_record.append(epoch)

        cliff_rmse, test_cliff_pred, test_cliff_true = calc_cliff_rmse(y_test_pred=test_pred, y_test=test_true, cliff_mols_test=data_1.cliff_mols_test)
        cliff_pred_list.append(test_cliff_pred)
        cliff_true_list.append(test_cliff_true)
        rmse_cliff.append(cliff_rmse)

        display = False
        del init_model
        torch.cuda.empty_cache()

    mean_rmsecliff = sum(rmse_cliff)/len(rmse_cliff)
    print('dataset_name:',  config_raw['dataset_name'])
    print('Cliff--rmse：', rmse_cliff, 'mean value：', mean_rmsecliff)

    # print(cliff_pred_list)
    # print(cliff_true_list)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    test_pred()