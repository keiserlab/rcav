# Import
import copy as cp
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from scipy.special import softmax
import scipy.stats as stats
from scipy.spatial.distance import cosine
import random
import math
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchUtils
import torch.optim as optim
from torchvision import models

import utils
import inception_mixup
import rcav
import rcav_utils
import TFMNIST
import struct
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    '''
    Note that all hyper parameters are set according to the experiment shown in paper figure 2. A number of additional hyperparameters for FMNIST and RCAV not set by the argparser can be found labelled by '#HYPERPARAMETER' in the code below.
    '''
    parser = argparse.ArgumentParser(description='Run RCAV on TFMNIST to reproduce figure 2 of paper')
    parser.add_argument('-t', '--train', type=bool, default=False, help='Whether to train or load trained model')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size if training')
    parser.add_argument('-n', '--n_permutations', type=int, default=500, help='Number of permutations drawn to approximate RCAV p-value')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train for')

    kwargs = parser.parse_args()
    train=kwargs.train
    load_saved_model=(not train)
    batch_size=kwargs.batch_size
    n_perm = kwargs.n_permutations
    epochs = kwargs.epochs
    
    
    fmnist_dir = './'
    texture_dir = './textures'
    save_dir=None
    counterfactual_aug=True
    
#################### Construct TFMNIST ###############################
    class_textures = {0:[1,6,4], 6:[2,5,9]}  #HYPERPARAMETER FOR TFMNIST
    for i in range(10):
        if not i in [0,6]: class_textures[i]=[0,3,7,8]
            
    if train:
        train_ims, train_inds, train_meta = [], [], []
        texturizer = TFMNIST.TexturedFMNIST(texture_dir = texture_dir, fmnist_dir=fmnist_dir)
        for cl in range(10):
            textures = class_textures[cl]
            data = texturizer.build_class(cl, train=True, texture_choices=textures, alpha=False, texture_rescale=False, texture_aug=True, aug_intensity=0.5)
            train_ims.append(data[0])
            train_inds.append(data[1])
            train_meta.append(data[2])

        train_im_list, train_label_list,train_ind_list, train_meta_list = [], [], [], []
        for i,images in enumerate(train_ims):
            train_im_list.extend(images)
            train_label_list.extend(len(images)*[i])
            train_meta_list.extend(train_meta[i])
            train_ind_list.extend(train_inds[i])

        train_im_array = np.stack(train_im_list).astype(np.uint8)
        train_label_array = np.array(train_label_list)
        train_ind_array = np.array(train_ind_list)
        train_texture_array = np.array([traindict['textures'][0] for traindict in train_meta_list])

        shuffle_inds_train = list(range(len(train_im_array)))
        random.shuffle(shuffle_inds_train)
        train_im_array = train_im_array[shuffle_inds_train]
        train_label_array = train_label_array[shuffle_inds_train]
        train_ind_array = train_ind_array[shuffle_inds_train]
        train_texture_array = train_texture_array[shuffle_inds_train]
        train_array_dict = {'X':train_im_array ,'Y':train_label_array ,'meta_inds':train_ind_array, 'meta_texts':train_texture_array}

        train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.RandomAffine(50, translate=(0.05,0.05), scale=(0.9,1.1), shear=(-10,10)), transforms.ToTensor()])
        train_set = utils.im_dataset(train_im_array, train_label_array, transform = train_transforms)
        train_loader = torchUtils.DataLoader(train_set, batch_size=batch_size, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=1, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None)

    val_ims, val_inds, val_meta = [], [], []
    texturizer = TFMNIST.TexturedFMNIST(texture_dir = texture_dir, fmnist_dir=fmnist_dir)
    for cl in range(10):
        textures = class_textures[cl]
        data = texturizer.build_class(cl, train=False, texture_choices=textures, alpha=False, texture_rescale=False, texture_aug=True, aug_intensity=0.5)
        val_ims.append(data[0])
        val_inds.append(data[1])
        val_meta.append(data[2])

    val_im_list, val_label_list,val_ind_list, val_meta_list = [], [], [], []
    for i,images in enumerate(val_ims):
        val_im_list.extend(images)
        val_label_list.extend(len(images)*[i])
        val_meta_list.extend(val_meta[i])
        val_ind_list.extend(val_inds[i])

    val_im_array = np.stack(val_im_list).astype(np.uint8)
    val_label_array = np.array(val_label_list)
    val_ind_array = np.array(val_ind_list)
    val_texture_array = np.array([valdict['textures'][0] for valdict in val_meta_list])

    shuffle_inds_val = list(range(len(val_im_array)))
    random.shuffle(shuffle_inds_val)
    val_im_array = val_im_array[shuffle_inds_val]
    val_label_array = val_label_array[shuffle_inds_val]
    val_ind_array = val_ind_array[shuffle_inds_val]
    val_texture_array = val_texture_array[shuffle_inds_val]

    val_array_dict = {'X':val_im_array ,'Y':val_label_array ,'meta_inds':val_ind_array, 'meta_texts':val_texture_array}
    val_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
    val_set = utils.im_dataset(val_im_array, val_label_array, transform = val_transforms)

    val_loader = torchUtils.DataLoader(val_set, batch_size=batch_size, shuffle=False, sampler=None,
                                   batch_sampler=None, num_workers=1, collate_fn=None,
                                   pin_memory=False, drop_last=False, timeout=0,
                                   worker_init_fn=None)
    
#################### Load or train model ###############################
    print('Loading model')
    
    #HYPERPARAMETERS MODEL TRAINING
    lr_factor = 0.2
    lr_patience = 10
    decay = 0.98
    stop_patience = 15

    model = inception_mixup.inception_v3(pretrained=True, aux_logits=False, manifold_mix='input', mixup_alpha=0.2, transform_input=False)
    model.fc = nn.Linear(2048, 10) #FMNIST has 10 classes
    model.Conv2d_1a_3x3 = utils.BasicConv2d(1, 32, kernel_size=3, stride=2) #FMNIST has only one channel
    model = model.cuda()

    BCE = nn.BCELoss()
    softmax = nn.Softmax(dim=1)
    criterion = lambda x,y: BCE(softmax(x),y)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=lr_factor, patience=lr_patience, verbose=True, threshold=-0.01) 
    decay_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay, last_epoch=-1)
    schedulers = {'decay':[decay_scheduler,None], 'plateau':[plateau_scheduler,'acc'],}

    runner = utils.RunNet(model, criterion, optimizer, 10,
                              schedulers=schedulers, save_dir=save_dir, mixup=True)
    if load_saved_model: model.load_state_dict(torch.load('TFMNIST_weights.pt'))
    if train:
        for epoch in tqdm(range(epochs)):
            print('Epoch ', epoch+1)
            runner.do_epoch(train_loader, train=True)
            print('train loss', runner.losses)
            runner.do_epoch(val_loader, train=False)
            runner.get_results('val')
            runner.schedule()
            print('val set performance', [(metric, runner.results['val'][metric][-1]) for metric in runner.results['val']])
            print()

#################### Compute ground truth softmax differences ###############################
    print('Computing ground truth concept sensitivity')
    if counterfactual_aug:
        #HYPERPARAMETERS RCAV
        benchmark_classes = [0]
        benchmark_class = benchmark_classes[0]
        target_class = 6
        textures_to_interp = [2,5,9]
        val_interp_list, val_baseline_list = [], []
        texturizer = TFMNIST.TexturedFMNIST(texture_dir = texture_dir, fmnist_dir=fmnist_dir)

        for i,ind in enumerate(val_ind_array):
            if val_label_array[i] in benchmark_classes:
                baseline_texture = val_texture_array[i]
                interp_texture = random.choice(textures_to_interp)
                val_interp_list.append(texturizer.get_textured_sample(ind, offset=(0,0), texture_choices=[baseline_texture, interp_texture], 
                                                randomize_textures=False, alpha=0.90, texture_rescale=False, texture_aug=False, aug_intensity=0)[0].astype(np.uint8))
                val_baseline_list.append(texturizer.get_textured_sample(ind, offset=(0,0), texture_choices=[baseline_texture, interp_texture], 
                                                randomize_textures=False, alpha=False, texture_rescale=False, texture_aug=False, aug_intensity=0)[0].astype(np.uint8))

        labels = len(val_interp_list)*[benchmark_class]
        val_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
        baseline_val_set = utils.im_dataset(val_baseline_list, labels, transform = val_transforms)
        interp_val_set = utils.im_dataset(val_interp_list, labels, transform = val_transforms)

        baseline_loader = torchUtils.DataLoader(baseline_val_set, batch_size=batch_size, shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=0, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None)
        interp_loader = torchUtils.DataLoader(interp_val_set, batch_size=batch_size, shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=0, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None)

        runner =  utils.RunNet(model, None, None, 2,
                            schedulers=dict(), save_dir=None)
        runner.do_epoch(interp_loader, False)
        runner.get_results('interp',format_only=True)
        interp_preds = cp.copy(runner.preds)
        interp_labels = cp.copy(runner.label_list)

        runner =  utils.RunNet(model, None, None, 2,
                            schedulers=dict(), save_dir=None)
        runner.do_epoch(baseline_loader, False)
        runner.get_results('baseline',format_only=True)
        baseline_preds = cp.copy(runner.preds)
        baseline_labels = cp.copy(runner.label_list)

#################### Construct CAV Concept Set ###############################
    print('Building CAV concept set')
    pos_cav_ims, neg_cav_ims = [], []
    texturizer = TFMNIST.TexturedFMNIST(texture_dir = texture_dir, fmnist_dir=fmnist_dir)
    #HYPERPARAMETERS RCAV
    pos_texture = [1,6,4]
    neg_texture = [2,5,9]
    classes = 10
    samples = 50
    label_to_cav_class = {0:'Spiral', 1:'Zigzag'}

    for cl in range(classes):
        data = texturizer.build_class(cl, num_samples=samples, train=False, texture_choices=pos_texture, alpha=False, texture_rescale=False, texture_aug=False, aug_intensity=0.5)
        pos_cav_ims.extend(data[0])

    for cl in range(classes):
        data = texturizer.build_class(cl, num_samples=samples, train=False, texture_choices=neg_texture, alpha=False, texture_rescale=False, texture_aug=False, aug_intensity=0.5)
        neg_cav_ims.extend(data[0])

    random.shuffle(pos_cav_ims)
    random.shuffle(neg_cav_ims)
    all_cav_ims = pos_cav_ims[:250]+neg_cav_ims[:250]
    all_cav_labels = 250*[1]+250*[0]
    all_cav_ims = np.stack(all_cav_ims).astype(np.uint8)
    all_cav_labels = np.array(all_cav_labels)

    cav_set = utils.im_dataset(all_cav_ims, all_cav_labels, transform = val_transforms)
    cav_loader = torchUtils.DataLoader(cav_set, batch_size=batch_size, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=0, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None)

#################### Run RCAV to get sensitivity scores ###############################
    print('Running RCAV')
    #HYPERPARAMETERS RCAV
    step_size = 10
    RCAV = True
    layers_to_test = ['Mixed_6a']
    concepts_to_test = [0]
    num_tests = 1
    ground_truth_diffs = np.array(interp_preds[:,target_class]-baseline_preds[:,target_class])

    model.eval()
    for layer in layers_to_test:
        print(layer)
        test = rcav.RCAV(model, layer, cav_set, baseline_val_set, baseline_loader, all_cav_labels, 
                                        num_classes=10, class_nums=[benchmark_class], target_class_num=target_class, multiple_tests_num=num_tests, TCAV=False)
        for pos_class in concepts_to_test:
            cav_score, significance = test.run(pos_class, n_random=n_perm, null_hypothesis='permutation', step_size=step_size, early_stop=False)
            pval = test.benchmark_sample_correlation(ground_truth_diffs)
            print('Image-level correlation has tau={0:.4f}, with un-adjusted p={1:.4f}'.format(test.trained_tau[0], pval)) #Note this raw p-value is only accurate down to p=1/(n_perm+1)
            fig = plt.figure(figsize=(14,7))
            plt.ylim(-0.001,0.012)
            p = sns.regplot(x=ground_truth_diffs, y=test.cav_diffs,  label='RCAV', scatter_kws={'alpha':0.5})
            p = p.get_figure()
            p.savefig('RCAV_fig2.png')