import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import rcav_utils
import scipy.stats as stats
import scipy.spatial as spatial
import random
import torch
from torch.nn import Softmax
from scipy.special import softmax
import copy as cp
import pickle
from tqdm import tqdm
import math

class RCAV():
    def __init__(self, model, layer_name, train_dataset, val_dataset, val_loader, concept_labels, num_classes, class_nums, target_class_num, multiple_tests_num=1, TCAV=False, **logistic_regression_kwargs):
        '''
        Args:
            model (pytorch model): trained model
            layer_name (string): layer choice
            train_dataset (torch.utils.data.Dataset): concept set consisting of (image, concept_label) pairs
            val_dataset (torch.utils.data.Dataset): dataset consisting of (image, label) pairs i.e. a subset of the val set corrresponding the train set on which the model was trained.
            val_loader (torch.utils.data.DataLoader): loader corresponding to val_dataset. Note that sampling order must be consistent across passes i.e. SHUFFLE=FALSE.
            concept_labels (np array): array of concept labels in the same order as used in train_dataset
            num_classes (int): number of classes
            class_nums (list of int): labels (classes) over which TCAV will be calculated
            target_class_num (int): label (class) number for which TCAV concept sensitivity will be calculated
            multiple_tests_num (int): Number of hypothesis tests carried out in this experiment. Used to define early stopping criterion for non-significance. Returned p-values are non-adjusted.
            TCAV (bool): Whether to calculate concept sensitivity using TCAV method or RCAV method. Default is RCAV.
        '''
        self.model = model
        self.layer_name = layer_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.concept_labels = concept_labels
        self.num_classes = num_classes
        self.class_nums = class_nums
        self.class_num = class_nums[0]
        self.target_class_num = target_class_num
        self.multiple_tests_num = multiple_tests_num
        self.TCAV = TCAV
        self.logistic_regression_kwargs = logistic_regression_kwargs
        self.step_size = None
        
        self._reset_fields(size_change=True)
        self.grads = None
        self.concept_acts = None
        
        self.random_cav_scores = []
        self.random_cav_accs = []
        self.random_cavs = []
        
        self.significance_dict = dict()
        self.model.eval()
        
    def _reset_fields(self, size_change):
        '''
        Resets fields between runs. Used if testing the same layer and target class, but on a different concept or with a different step size.
        '''
        self.sample_cav = None
        self.cav_score = None
        self.cav_scores  = []
        self.cavs  = []
        self.cav_accs  = []
        if size_change: 
            self.random_cav_scores = []
            self.random_latent_aug_preds = []
        
    def _get_acts_subset(self, pos_concept_num, sample_size=None, random_concept=True, bootstrap=True, randomized_concept_labels=None):
        '''
        Gets activations for either the subset excluding pos_concept_num (if random_concept=True), otherwise the subset of only pos_concept_num.
        
        Args:
            pos_concept_num: int, the concept number for this run of TCAV
            sample_size: int, the bootstrap sample size to draw
            random_concept: Bool, whether to sample from pos_concept or random concepts
            bootstrap: Bool, whether to use the whole sample or resample by boostrapping
            randomized_concept_labels: list of int, an alternative to self.concept_labels to use for generating null distribution
        '''
        if sample_size is None: 
            if random_concept: sample_size = sum(self.concept_labels!=pos_concept_num)
            else: sample_size = sum(self.concept_labels==pos_concept_num)
        
        if randomized_concept_labels: weight = [0 if random_concept^(label!=pos_concept_num) else 1 for label in randomized_concept_labels]
        else: weight = [0 if random_concept^(label!=pos_concept_num) else 1 for label in self.concept_labels]
        tot = sum(weight)
        weight = [indicator/tot for indicator in weight]
        act_inds = np.random.choice(np.arange(len(self.concept_labels)), size=sample_size, replace=bootstrap, p=weight)
        acts_subset = [self.concept_acts[ind] for ind in act_inds]
        if random_concept: labelled_acts_subset = [[act[0], 0] for act in acts_subset]
        else: labelled_acts_subset = [[act[0], 1] for act in acts_subset]
        return labelled_acts_subset

    def _get_cav(self, acts,):
        cav = CAV()
        cav.train(acts, **self.logistic_regression_kwargs)
        return cav

    def _get_TCAV_score_significance(self, null_hypothesis, early_stop=False):
        '''
        Computes TCAV sensitivity scores
        '''
        if early_stop: raise NotImplementedError()
        if self.grads is None: self.grads = rcav_utils.get_grads(self.model, self.layer_name, self.val_dataset, self.class_nums, self.target_class_num, self.num_classes)
        self.cav_scores = [sum([1 if np.dot(grad,cav.vec)>0 else 0 for grad in self.grads])/len(self.grads) for cav in self.cavs]
        self.cav_score = self.cav_scores[0]
        if null_hypothesis=='ttest_onesamp': 
            self.significance = stats.ttest_1samp(self.cav_scores, 0.5)
            self.significance = (self.significance[0], min(1,self.significance[1]))
            self.significance_dict[self.pos_concept_num] = cp.deepcopy((self.cav_score, self.significance))
            return self.cav_score, self.significance
        elif null_hypothesis=='permutation' or null_hypothesis=='gaussian_null':
            if null_hypothesis=='permutation':
                for i in tqdm(range(self.n_random), desc='Sampling Null Distribution'):
                    randomized_concept_labels = [act[1] for act in self.concept_acts]
                    random.shuffle(randomized_concept_labels)
                    pos_acts = self._get_acts_subset(self.pos_concept_num, sample_size=self.sample_size, randomized_concept_labels=randomized_concept_labels, bootstrap=False, random_concept=False)
                    rand_acts = self._get_acts_subset(self.pos_concept_num, sample_size=self.sample_size, randomized_concept_labels=randomized_concept_labels, bootstrap=False)
                    new_cav = self._get_cav(rand_acts+pos_acts)
                    self.random_cavs.append(new_cav)
                    self.random_cav_accs.append(new_cav.acc)
            self.random_cav_scores = [sum([1 if np.dot(grad,cav.vec)>0 else 0 for grad in self.grads])/len(self.grads) for cav in self.random_cavs]
            cav_from_50 = [np.abs(score-0.5) for score in self.cav_scores][0]
            random_cav_from_50 = [np.abs(score-0.5) for score in self.random_cav_scores]
            self.significance = (cav_from_50, min(1,sum([int(random_score>cav_from_50) for random_score in random_cav_from_50])/len(random_cav_from_50)))
            self.significance_dict[self.pos_concept_num] = cp.deepcopy((self.cav_score, self.significance))
            return self.cav_score, self.significance
        else:
            raise ValueError('null_hypothesis must be in ["ttest_onesamp","permutation","gaussian_null"]')
            
    def _get_RCAV_score_significance(self, null_hypothesis, early_stop=True):
        '''
        Computes RCAV sensitivity scores
        '''
        if null_hypothesis=='ttest_onesamp':
            assert early_stop==False
            
        # Compute RCAV sensitivity scores for the actual CAV
        pair_softmax = Softmax(dim=-1)
        self.baseline_preds, self.cav_latent_aug_preds = [], []
        with torch.no_grad():
            self.model.latent_aug = None
            for batch in self.val_loader:
                if type(batch)==dict: inputs = batch['image'].cuda()
                else: inputs = batch[0].cuda()
                self.baseline_preds.append(self.model(inputs, aug=None).cpu())
            self.baseline_preds = softmax(np.vstack(torch.cat(self.baseline_preds).numpy()), axis=1)
        with torch.no_grad():
            aug_tensor = torch.Tensor(self.sample_cav.vec.reshape(self.acts_dimensions))
            aug_tensor = self.step_size*aug_tensor/torch.norm(aug_tensor)
            self.model.latent_aug = self.layer_name
            for batch in self.val_loader:
                if type(batch)==dict: inputs = batch['image'].cuda()
                else: inputs = batch[0].cuda()
                aug_batch = torch.cat(inputs.shape[0]*[aug_tensor]).cuda()
                self.cav_latent_aug_preds.append(self.model(inputs, aug=aug_batch).cpu())
            self.cav_latent_aug_preds = softmax(np.vstack(torch.cat(self.cav_latent_aug_preds).numpy()), axis=1)
            
        self.cav_scores = [sum([1 if self.cav_latent_aug_preds[p][self.target_class_num]>=baseline_pred[self.target_class_num] else 0 for p,baseline_pred in enumerate(self.baseline_preds)])/len(self.baseline_preds) 
                                   for cav in self.cavs]
        self.cav_score = self.cav_scores[0]
        
        # Compute RCAV sensitivity scores for null set CAVs
        null_threshold = math.ceil(0.05*self.n_random/self.multiple_tests_num)
        cav_from_50 = [np.abs(score-0.5) for score in self.cav_scores][0]
        self.model.latent_aug = self.layer_name
        null_count = 0
        randomized_concept_labels = [act[1] for act in self.concept_acts]
        for i in tqdm(range(self.n_random), desc='Sampling Null Distribution'):
            if len(self.random_cav_scores) < i+1 or null_hypothesis=='ttest_onesamp':
                if null_hypothesis=='permutation' and len(self.random_cavs)<i+1:
                    random.shuffle(randomized_concept_labels)
                    pos_acts = self._get_acts_subset(self.pos_concept_num, sample_size=self.sample_size, randomized_concept_labels=randomized_concept_labels, bootstrap=False, random_concept=False)
                    rand_acts = self._get_acts_subset(self.pos_concept_num, sample_size=self.sample_size, randomized_concept_labels=randomized_concept_labels, bootstrap=False)
                    new_cav = self._get_cav(rand_acts+pos_acts)
                    self.random_cavs.append(new_cav)
                    self.random_cav_accs.append(new_cav.acc)
                if null_hypothesis=='ttest_onesamp': cav = self.cavs[i]
                else: cav = self.random_cavs[i]
                aug_tensor = torch.Tensor(cav.vec.reshape(self.acts_dimensions))
                aug_tensor = self.step_size*aug_tensor/torch.norm(aug_tensor)
                self.latent_aug_preds = []
                with torch.no_grad():
                    for batch in self.val_loader:
                        if type(batch)==dict: inputs = batch['image'].cuda()
                        else: inputs = batch[0].cuda()
                        aug_batch = torch.cat(inputs.shape[0]*[aug_tensor]).cuda()
                        self.latent_aug_preds.append(self.model(inputs, aug=aug_batch).cpu())
                    self.latent_aug_preds = softmax(np.vstack(torch.cat(self.latent_aug_preds).numpy()), axis=1)
                self.random_latent_aug_preds.append(cp.deepcopy(self.latent_aug_preds))
                self.random_cav_scores.append(sum([1 if self.latent_aug_preds[p][self.target_class_num]>baseline_pred[self.target_class_num] else 0 for p,baseline_pred in 
                                               enumerate(self.baseline_preds)])/len(self.baseline_preds))
            if early_stop:
                null_check = np.abs(self.random_cav_scores[i]-0.5)>=cav_from_50
                if null_check: null_count = null_count+1
                if null_count>=null_threshold: 
                    random_cav_from_50 = [np.abs(score-0.5) for score in self.random_cav_scores]
                    self.significance = (cav_from_50, min(1,sum([int(random_score>=cav_from_50) for random_score in random_cav_from_50])/len(random_cav_from_50)))
                    self.significance_dict[self.pos_concept_num] = cp.deepcopy((self.cav_score, self.significance))
                    return self.cav_score, self.significance
                
        if null_hypothesis in ['permutation', 'gaussian_null']:
            random_cav_from_50 = [np.abs(score-0.5) for score in self.random_cav_scores]
            self.significance = (cav_from_50, min(1,sum([int(random_score>=cav_from_50) for random_score in random_cav_from_50])/len(random_cav_from_50)))
            self.significance_dict[self.pos_concept_num] = cp.deepcopy((self.cav_score, self.significance))
            return self.cav_score, self.significance
        elif null_hypothesis=='ttest_onesamp': 
            self.significance = stats.ttest_1samp(self.random_cav_scores, 0.5)
            self.significance_dict[self.pos_concept_num] = cp.deepcopy((self.cav_score, self.significance))
            return self.cav_score, self.significance
        else:
            raise ValueError('null_hypothesis must be in ["ttest_onesamp","permutation", "gaussian_null"]')
            
    def benchmark_sample_correlation(self, ground_truth_score_delta, hypothesis_test=stats.kendalltau):
        '''
        Given ground truth for image-level concept sensitivity, computes performance metrics
        Args:
            ground_truth_score_delta: (np array) ground truth softmax differences
            hypothesis_test: function to use for hypothesis test statistic. this function will _not_ be used for the p-value, only the statistic.
        Returns:
            Hypothesis test p-value
        '''
        if not self.TCAV:
            self.cav_diffs = [self.cav_latent_aug_preds[p][self.target_class_num]-baseline_pred[self.target_class_num] for p,baseline_pred in enumerate(self.baseline_preds)]
            self.trained_tau = hypothesis_test(ground_truth_score_delta, self.cav_diffs)
            self.random_tau = []
            for aug_preds in self.random_latent_aug_preds:
                diffs = [aug_preds[p][self.target_class_num]-baseline_pred[self.target_class_num] for p,baseline_pred in enumerate(self.baseline_preds)]
                self.random_tau.append(hypothesis_test(ground_truth_score_delta, diffs))
        else:
            self.cav_diffs = np.array([spatial.distance.cosine(self.sample_cav.vec,grad) for grad in self.grads])
            self.trained_tau = hypothesis_test(ground_truth_score_delta, self.cav_diffs)
            self.random_tau = []
            for cav in self.random_cavs:
                rand_cav_diffs = np.array([spatial.distance.cosine(cav.vec,grad) for grad in self.grads])
                self.random_tau.append(hypothesis_test(ground_truth_score_delta, rand_cav_diffs))
            
        return sum([1 for rt in self.random_tau if self.trained_tau[0]<=rt[0]])/len(self.random_tau)
    
    def save(self, save_dir, save_prefix):
        '''
        Note that saving clears all high memory usage fields.
        '''
        self.cavs = [self.cavs[0]]
        self.val_loader = None
        self.val_dataset = None
        self.grads = None
        self.model = None
        self.concept_acts = []
        self.random_cavs = []
        self.train_dataset=None
        save_loc = save_dir+save_prefix+'_{0}.pkl'.format(self.layer_name)
        with open(save_loc, 'wb') as f: pickle.dump(self.__dict__,f)

    
    def run(self, pos_concept_num, sample_size=None, n_random=500, step_size=10, null_hypothesis='permutation', early_stop=True):
        '''
        Builds CAVs and calculates scores using the subset of val_dataset corresponding to class_num. If doing multiple tests on the same layer you can use run() without re-initializing the class instance.
        Args:
            sample_size: int or None, size of bootstrap and permutation set null CAV training sets. If None defaults to the size of the dataset.
            n_random: int, number of boostrap or permutation set null samples to use
            step_size: float, step size for RCAV
            null_hypothesis: str in ['ttest_onesamp','permutation','gaussian_null'], defines the hypothesis test used
            early_stop: bool, whether to early stop when non-significance level is reached for gaussian null or permutation test.
        Returns:
            sensitivity score, (test statistic, p-value)
        '''
        self._reset_fields(size_change=(self.step_size!=step_size))
        self.n_random = n_random
        self.step_size = step_size
        self.pos_concept_num = pos_concept_num
        self.sample_size = sample_size
        if self.concept_acts is None: self.concept_acts, self.acts_dimensions = rcav_utils.get_acts(self.model, self.layer_name, self.train_dataset, self.concept_labels)
            
        # First get cav score on given samples
        pos_acts = self._get_acts_subset(pos_concept_num, sample_size=sample_size, random_concept=False, bootstrap=False)
        rand_acts = self._get_acts_subset(pos_concept_num, sample_size=sample_size, bootstrap=False)
        self.sample_cav = self._get_cav(rand_acts+pos_acts)
        self.cavs.append(self.sample_cav)
        self.cav_accs.append(self.sample_cav.acc)
        
        if null_hypothesis in ['ttest_onesamp']:
            for _ in tqdm(range(n_random), desc='Training Random CAVs'):
                pos_acts = self._get_acts_subset(pos_concept_num, sample_size=sample_size, random_concept=False)
                rand_acts = self._get_acts_subset(pos_concept_num, sample_size=sample_size)
                self.cavs.append(self._get_cav(rand_acts+pos_acts))
                self.cav_accs.append(self.cavs[-1].acc)
                
        if null_hypothesis=='gaussian_null':
            self.random_cavs = [CAV() for i in range(n_random)]
            for cav in self.random_cavs: cav.vec = np.random.normal(size=self.sample_cav.vec.shape)
            
        if self.cav_accs[0] < 0.8: print('Warning: the CAV accuracy of {0} is low, so TCAV results may  not be meaningful'.format(self.cav_accs[0]))
        if self.TCAV: return self._get_TCAV_score_significance(null_hypothesis, early_stop=early_stop)
        else: return self._get_RCAV_score_significance(null_hypothesis, early_stop=early_stop)
        
        
class CAV():
    def __init__(self,):
        '''        
        '''
        self.vec = None
        self.acc = None
        self.class_balance = None

    def train(self, acts, **logistic_regression_kwargs):
        '''
        Args:
            acts: list of [act,concept] pairs.
            logistic_regression_kwargs: hyper_parameters for linear model
        '''
        # convert acts into arrays for training
        X,Y = [],[]
        for act,concept in acts:
            X.append(act)
            Y.append(concept)
        X,Y = np.array(X),np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)
        if 'alpha' not in logistic_regression_kwargs: lm = linear_model.SGDClassifier(alpha=.01, max_iter=1000, tol=1e-3, **logistic_regression_kwargs) 
        else: lm = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, **logistic_regression_kwargs)
        lm.fit(X_train, Y_train)
        self.acc = lm.score(X_test,Y_test)
        self.vec = lm.coef_[0]
        self.class_balance = np.unique(Y,return_counts=True)[1][1]/len(Y)