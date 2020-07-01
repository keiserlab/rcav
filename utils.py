import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.optim as optim
from torchvision import models
import numpy as np
import pickle
from scipy.special import softmax
import sklearn
import copy as cp


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
    
class im_dataset(utils.Dataset):
    def __init__(self, X, Y, transform=None):
        assert len(X)==len(Y)
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        img = self.X[index]
        labels = self.Y[index]

        if self.transform: return self.transform(img), labels
        else: return img, labels

    def __len__(self):
        return len(self.X)
    
class RunNet():
    def __init__(self, model, criterion, optimizer, n_classes,
                     schedulers=dict(), save_dir=None, mixup=False):
        '''
        schedulers: dict, of the form 'name':[object, metric] where metric may be None
        '''
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_classes = n_classes
        
        self.schedulers = schedulers
        self.results = dict()
        self.save_dir = save_dir
        self.stop = False # used to flag early stopping i.e. if 'early_stop' in self.schedulers.keys(): 
        
        self.mixup = mixup
        
    def do_batch(self, inputs, labels, train, other=None):
        '''
        '''
        inputs, labels = inputs.cuda(), labels.cuda()
        if not self.mixup or not train:
            outputs = self.model(inputs)
            return outputs, labels
        else:
            outputs, mix_info = self.model(inputs, target=labels)
            y_a, y_b, lam = mix_info
            if len(y_a.shape)<2:
                zerosa = torch.zeros(len(y_a),self.n_classes).cuda()
                zerosa = zerosa.scatter(1,y_a.reshape(-1,1),1)
                zerosb = torch.zeros(len(y_b),self.n_classes).cuda()
                zerosb = zerosb.scatter(1,y_b.reshape(-1,1),1)
                mixed_labels = lam*zerosa+(1-lam)*zerosb
            else: 
                mixed_labels = lam*y_a+(1-lam)*y_b
            return outputs, mixed_labels
            
    def do_epoch(self, loader, train):
        '''
        train: bool, whether or not to keep track of and apply gradients
        '''
        self.preds, self.label_list, self.losses = [], [], []
        # Loop through loader
        if train:
            self.model.train()
            for i, data in enumerate(loader):
                self.optimizer.zero_grad()
                if type(data)==dict:
                    outputs, labels = self.do_batch(data['image'], data['labels'], train)
                else:
                    outputs, labels = self.do_batch(data[0], data[1], train)
                if len(labels.shape)==1: loss = self.criterion(outputs, labels)
                else: loss = self.criterion(outputs, labels) #Note this labels shape is non-standard
                loss.backward()
                self.losses.append(loss.item())
                self.optimizer.step()
                self.preds.append(outputs.detach())
                self.label_list.append(labels)
            self.losses = np.mean(self.losses)
        else:
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(loader):
                    if type(data)==dict:
                        outputs, labels = self.do_batch(data['image'], data['labels'], train)
                    else:
                        outputs, labels = self.do_batch(data[0], data[1], train)
                    self.preds.append(outputs)
                    self.label_list.append(labels)
        
    def get_results(self, name, format_only=False):
        '''
        name: str, one of 'train', 'val', 'test', or a variant thereof
        '''
        if not name in self.results.keys(): self.results[name] = dict()
        self.label_list = np.vstack(torch.cat(self.label_list).cpu().numpy())
        self.preds = softmax(np.vstack(torch.cat(self.preds).cpu().numpy()), axis=1)
        if format_only: return
        if len(self.label_list.shape)==1:
            epoch_results = get_performance_metrics(num_classes=self.n_classes,preds=self.preds,label_list=self.label_list)
        else:
            epoch_results = get_performance_metrics(num_classes=self.n_classes,preds=self.preds,label_list=self.label_list[:,0])
        if not name in self.results: self.results[name] = dict()
        for result in epoch_results.keys():
            if not result in self.results[name]: self.results[name][result] = []
            self.results[name][result].append(epoch_results[result])
        
    def schedule(self):
        # Apply schedulers
        if 'decay' in self.schedulers.keys():
            self.schedulers['decay'][0].step()
        if 'plateau' in self.schedulers.keys():
            self.schedulers['plateau'][0].step(self.results['val'][self.schedulers['plateau'][1]][-1])
        if 'early_stop' in self.schedulers.keys():
            self.stop = self.schedulers['early_stop'][0](self.results['val'][self.schedulers['early_stop'][1]][-1], self.model)
            
    def save(self, fold):
        torch.save(self.model.state_dict(), self.save_dir+'/'+'fold'+str(fold))
        with open(self.save_dir+'/'+"results"+str(fold)+".pkl", "wb") as file:
            pickle.dump(self.results,file)
            

def get_performance_metrics(num_classes, preds, label_list, metrics=['acc','auprc','auroc','log_loss'], rounding=4):
    '''
    num_classes: integer
    metrics: list of strings. subset of ['acc','auprc','auroc','log_loss'] 
    '''

    results = dict()
    if 'auprc' in metrics or 'auroc' in metrics:
        if num_classes>2:
            relevant_classes = list(np.unique(label_list))
            Y_byclass = [np.array([[0,1] if row==l else [1,0] for row in label_list]) for l in range(num_classes)]
            pred_byclass = [cp.deepcopy(preds) for i in range(num_classes)]
            for j in range(num_classes):
                other = np.sum(pred_byclass[j][:,[i for i in range(num_classes) if i!=j]],axis=1)
                pred_byclass[j][:,1] = pred_byclass[j][:,j]
                pred_byclass[j][:,0] = other
                pred_byclass[j] = pred_byclass[j][:,[0,1]]
                
    for metric in metrics:
        if metric=='acc': results[metric] = round(sklearn.metrics.accuracy_score(label_list,np.argmax(preds,axis=1)),rounding)
        elif metric=='auprc': 
            if num_classes>2: results[metric] = round(np.mean([sklearn.metrics.average_precision_score(Y_byclass[cl],pred_byclass[cl]) for cl in range(num_classes) if cl in relevant_classes]),rounding)
            else: results[metric] = round(sklearn.metrics.average_precision_score(label_list,preds[:,1]),rounding)
        elif metric=='auroc':
            if num_classes>2: results[metric] = round(np.mean([sklearn.metrics.roc_auc_score(Y_byclass[cl],pred_byclass[cl]) for cl in range(num_classes) if cl in relevant_classes]),rounding)
            else: results[metric] = round(sklearn.metrics.roc_auc_score(label_list,preds[:,1]),rounding)
        elif metric=='log_loss':
            sklearn.metrics.log_loss(label_list,preds,labels=np.array(list(range(num_classes))))
    return results