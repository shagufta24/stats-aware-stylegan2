import numpy as np
import torch
import sys
sys.path.append('/home/nkamath5/miniconda3/pkgs/torch-two-sample-0.1-py36h39e3cac_0/lib/python3.6/site-packages')  
# explicitly added to path above since this package wasn't getting found during execution
import torch_two_sample as tst  # edited
import torch.distributed as dist
import torch.nn.functional as F
import os
import random
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pytorch_generative.models import kde
import torch.nn as nn
from scipy import stats
from scipy.stats import ks_2samp

import pyro
import pyro.distributions as pyro_dist

def public_features(features, data):
    # function copied from loss.py
    all_features = False
    if 'all' in  features:
        all_features = True
    final_features = [] # keep appending relevant features to this list & convert to tensor at the end
    data          = torch.squeeze(data).float()  # data is already expected to be in float, but leaving the .float() here
    if all_features or ('areas' in  features):
        phantom_masks = (data > -0.6471).float()  # [(45/127.5)-1] = -0.6471
        # gradients are zero at all places where a step function is differentiable, hence error signal is not passed to gen images via this route
        areas         = torch.sum(phantom_masks, dim=(1,2)) #.unsqueeze(1)
        final_features.append(areas)
    if all_features or ('fg_ratios' in  features) or ('fat_areas' in  features):
        fat           = (data >= -0.6471) * (data < -0.0588)  # [(120/127.5)-1] = -0.0588
        fat_areas     = torch.sum(fat, dim=(1,2)) #.unsqueeze(1)
        if all_features or ('fat_areas' in  features): final_features.append(fat_areas)
    if all_features or ('fg_ratios' in  features) or ('gln_areas' in  features):
        gln           = (data >= -0.0588) * (data < 0.7725)  # [(226/127.5)-1] = 0.7725
        gln_areas     = torch.sum(gln, dim=(1,2)) #.unsqueeze(1)
        if all_features or ('gln_areas' in  features): final_features.append(gln_areas)
    if all_features or ('fg_ratios' in  features):
        fg_ratios     = torch.log10(((fat_areas + 0.00012)/ (gln_areas + 0.0001)) + 1)
        final_features.append(fg_ratios)
        # 1 added since it protects from NaN values for low values inside log, without affecting gradient a lot (check graphically for clarity)
    N = len(data)  # alternatively N = data.shape[0], i.e. batch size
    data          = data.reshape(N, -1)
    if all_features or 'means' in  features or 'skewnesses' in  features or 'kurtoses' in  features or 'balances' in  features:
        means         = torch.mean(data, dim=1) #.unsqueeze(1)
        if all_features or ('means' in  features): final_features.append(means)
    if all_features or 'stds' in  features or 'skewnesses' in  features or 'kurtoses' in  features:
        stds          = torch.std(data, dim=1) #.unsqueeze(1)
        if all_features or ('stds' in  features): final_features.append(stds)
    if all_features or 'skewnesses' in  features:
        skewnesses    = (torch.mean((data - means.unsqueeze(-1))**3, dim=1) / stds.squeeze()**3) #.unsqueeze(1)
        final_features.append(skewnesses)
    if all_features or 'kurtoses' in  features:
        kurtoses      = ((torch.mean((data - means.unsqueeze(-1))**4, dim=1) / stds.squeeze()**4) - 3.0) #.unsqueeze(1)
        final_features.append(kurtoses)
    if all_features or 'balances' in  features:
        balances = (torch.quantile(data, 0.7, dim=1) - means) / (means - torch.quantile(data, 0.3, dim=1))
        final_features.append(balances)
                
    return torch.stack(final_features, dim=1).squeeze(1)


def sample_images(directory, b):
    # List all PNG files in the directory
    all_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    # Randomly sample 'b' files
    sampled_files = random.sample(all_files, b)
    images = [Image.open(os.path.join(directory, file)) for file in sampled_files]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = [torch.tensor(np.array(image)).to(device) for image in images]
    # Stack all tensors to form a single tensor
    images = torch.stack(images)
    images = (images - 127.5) / 127.5
    return images

def plot_loss_violins(data, batch_sizes, num_cosines_list, M_list, loss_name, k=None, save_path=None):
    """
    Plots violin plots for loss values across different batch sizes.

    :param data: A list of 1D numpy arrays. list dim is batch_size dim & array dim are the loss values for that batch_size
    :param batch_sizes: A list of batch sizes corresponding to the outer dimension of 'data'.
    :param num_cosines_list: A list of number of cosines corresponding to the outer dimension of 'data'.
    :param save_path: File path to save the plot image. If None, the plot is not saved.
    """
    # Check if the batch sizes list matches the data's first dimension, else it is some transpose of the data
    if len(data) != len(batch_sizes) or len(data) != len(num_cosines_list):
        raise ValueError("Length of batch_sizes and num_cosines_list must match the first dimension of data.")

    #breakpoint()
    plot_data = []
    for i, (batch_size, num_cosines) in enumerate(zip(batch_sizes, num_cosines_list)):
        for loss in data[i]:
            plot_data.append({'Config': (batch_size, num_cosines), 'Loss': loss})

    df = pd.DataFrame(plot_data)
    #breakpoint()

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Config', y='Loss', data=df)

    last_slash_index = save_path.rfind('/') # # Find the position of the last '/'
    ##png_index = save_path.rfind('.png') # Find the position of '.png'
    ##dir_part = save_path[last_slash_index + 1:png_index] # Extract the part of the string between the last '/' and '.png'
    dir_part = save_path[last_slash_index + 1:]
    if loss_name == 'kNN':
        plt.title(dir_part + f'\n{k}-NN tst Loss dispersion across batch sizes')
    else:
        plt.title(dir_part + f'\n {loss_name} Loss dispersion across batch sizes')
    
    plt.xlabel('Batch Size, Num of Cosines')
    
    if loss_name != 'Bhatt' and loss_name != 'KS':
        plt.ylim(-2, 15)

    # Add 'M' value annotations to each violin plot
    for i, M_value in enumerate(M_list):
        # Position of the text in the middle of each violin plot
        x_position = i
        y_position = ax.get_ylim()[1]
        plt.text(x_position, y_position, f'M={M_value}', horizontalalignment='center', verticalalignment='top', fontsize=10, color='blue')
    
    if save_path:
        save_path = save_path +'.png'  
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

loss_choices = ['FR', 'kNN', 'Bhatt', 'KL', 'KS', 'Bhatt2000']
parser = argparse.ArgumentParser(description='Examine different losses over various batch sizes multiple times')
parser.add_argument('loss', choices=loss_choices, help='Select one loss from FR or kNN')
parser.add_argument('--k', type=int, default=1, help='k for kNN')
parser.add_argument('--num_cosines', type=int, default=200, help='num_cosines; 0 means no pca+cosine logic; def 200')
#parser.add_argument('--nocosine', action='store_true', help='Include this flag to set it to True')
args = parser.parse_args()

cuda = torch.cuda.is_available()
print("Cuda is available: ", cuda) 

#dirs = ['00029-128-auto1-gamma2-batch64-kimg000302-125K', 
#        '00029-128-auto1-gamma2-batch64-kimg002721-125K', 
#        '00029-128-auto1-gamma2-batch64-kimg008467-125K', 
#        '00029-128-auto1-gamma2-batch64-kimg014212-125K']
#dirs = ['00030-128-auto1-gamma2-batch64-kimg000302-125K',
#        '00030-128-auto1-gamma2-batch64-kimg002721-125K', 
#        '00030-128-auto1-gamma2-batch64-kimg008467-125K']
dirs = ['00030-128-auto1-gamma2-batch64-kimg000302-125K', 
        '00030-128-auto1-gamma2-batch64-kimg002721-125K', 
        '00030-128-auto1-gamma2-batch64-kimg008467-125K', 
        '00029-128-auto1-gamma2-batch64-kimg000302-125K', 
        '00029-128-auto1-gamma2-batch64-kimg002721-125K', 
        '00029-128-auto1-gamma2-batch64-kimg008467-125K', 
        '00029-128-auto1-gamma2-batch64-kimg014212-125K']
dirs = [dirs[i] for i in range(0,2)]
loss, k = args.loss, args.k
#num_cosines = args.num_cosines
#b_list = [32, 128, 1024] 
#b_list = [4096, 8500]
#num_cosines_list = [(b*b)//2 if b<4000 else (b*b)//4 for b in b_list]
b_list = [1024] * 3
num_cosines_list = [10000, 1000, 200] # [int(b_list[i]*b_list[i]*(0.2**i)) for i in range(1,3)] #
#num_cosines_list.reverse()
for dir in dirs:
    print(f"dir is {dir}")
    gen_directory = '/data/nkamath5/stats_aware_gans/aapm_generated_images/' + dir
    real_directory = '/shared/rsaas/nkamath5/train_128'

    loss_values = [] # will be a list of lists
    M_list = [] # number of monte carlo runs
    use_pca_cosine_logic = True if any(num_cosines_list) else False
    for (batch_size, num_cosines) in zip(b_list, num_cosines_list):
        M = 300 if batch_size >= 1024 else 500
        #M = 20 if batch_size >= 4096 else M
        M_list.append(M)
        print(f'b is {batch_size}, num_cosines is {num_cosines}, use_pca_cosine_logic is {use_pca_cosine_logic} and M is {M}')

        loss_values_for_a_batch_size = []
        for i in tqdm(range (M), desc=f"Loss {args.loss} Monte Carlo iter for batch_size {batch_size} & cos {num_cosines}"):
            
            gen_img = sample_images(gen_directory, batch_size)
            real_img = sample_images(real_directory, batch_size)
            
            gen_vectors = public_features('all', gen_img)
            real_vectors = public_features('all', real_img)
    
            if loss in loss_choices:
                #assert num_cosines <= batch_size*batch_size #
                alphas = [2e-5]  # relevant for kNN & FR tests only
            
                if use_pca_cosine_logic == True:
                    _, _, V = torch.pca_lowrank(real_vectors, q=real_vectors.shape[1], center=True) 
                    # added the q parameter since by default it would have been 6 & we want it to be overestimated: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
                    # matmul(A, V[:, :k]) projects data to the first k principal components
                    real_vectors = torch.matmul(real_vectors, V[:, :real_vectors.shape[1]]) # pca projections
                    gen_vectors = torch.matmul(gen_vectors, V[:, :real_vectors.shape[1]]) # projecting onto p components of reals
                    # real_vectors & gen_vectors must now be of shape (batch_size_per_gpu x k)
                    
                    #sample 2 random vectors from reals & find their cosine similarities
                    # create a (2x batch_size) tensor; row 0 are indices for first vectors, row 1 for second in cos sim calculation
                    random_matching = torch.randint(low=0, high=real_vectors.shape[0], size=(2, num_cosines)) # will give ints with replacement!
                    real_vecs_set_1 = real_vectors[random_matching[0]]  # size (num_cosines, k)
                    real_vecs_set_2 = real_vectors[random_matching[1]]
                    cossim_reals = F.cosine_similarity(real_vecs_set_1, real_vecs_set_2, dim=1, eps=1e-8)
    
                    random_matching = torch.randint(low=0, high=real_vectors.shape[0], size=(2,  num_cosines))
                    real_vecs_set_1 = real_vectors[random_matching[0]]
                    gen_vecs_set_2 = gen_vectors[random_matching[1]]
                    cossim_mixed = F.cosine_similarity(real_vecs_set_1, gen_vecs_set_2, dim=1, eps=1e-8)
    
                    # cosine similarity vectors become our final real & gen vectors.
                    real_vectors = cossim_reals.unsqueeze(1) # to make 1-D tensor into 2-D tensor for purposes of code in two_sample_test
                    gen_vectors = cossim_mixed.unsqueeze(1)
    
                def get_log_prob(vectors):
                    """ # inputs: vectors must be n x d
                        # returns: estimator fitted to vectors using kde 
                        # estimator.forward() will return log probs
                        Currently we have two ways of doing kde:-
                            1. Using kde in https://github.com/EugenHotaj/pytorch-generative
                            2. Using the pyro library https://pyro.ai/examples/index.html
                        Both have a space complexity of O(training_vector_size * test_vector_size)
                        where training_vectors are those at which kernels are fitted and 
                        testing_vectors are those where prob/log_prob are estimated"""
                    scotts_factor = num_cosines**(-1.0/(4+vectors.shape[1])) # heuristic for bandwidth computation
                    mean_vector   = vectors.mean(dim=0) # mean for each dimension/ col, so taken across rows
                    normalized_vectors = vectors - mean_vector.unsqueeze(0)  # nxd - 1xd
                    covariance    = torch.mm(torch.t(normalized_vectors), normalized_vectors) / (vectors.shape[0]-1) # (dxn) @ (nxd)
                    bandwidth     = covariance*(scotts_factor**2)
                    log_prob_estimator = kde.KernelDensityEstimator(vectors, kde.GaussianKernel(bandwidth))
                    #log_prob_estimator = pyro_dist.Normal(vectors.squeeze(), bandwidth)
                    #breakpoint()
                    return log_prob_estimator
                    
                if use_pca_cosine_logic:
                    # since the datapoints being compared for similar distribution are now num_cosines in number
                    if   loss == 'FR':
                        loss_tst_fn = tst.statistics_diff.SmoothFRStatistic(num_cosines, num_cosines, cuda=cuda, compute_t_stat=True) 
                    
                    elif loss == 'kNN':
                        loss_tst_fn = tst.statistics_diff.SmoothKNNStatistic(num_cosines, num_cosines, cuda, k, compute_t_stat=True)
                    
                    elif loss == 'Bhatt' or loss == 'Bhatt2000' or loss[0:5] == 'Bhatt':
                        # we will use our data points (i.e. cosines) to estimate its PDF using KDE and 
                        # then use the probabilities to copmute the Bhattacharyya distance
                        # ref https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py
                        gen_vectors, real_vectors = gen_vectors.to('cpu'), real_vectors.to('cpu')
                        concatted_vectors = torch.cat((gen_vectors, real_vectors), dim=0)
                        log_prob_gen_estimator  = get_log_prob(gen_vectors)
                        log_prob_real_estimator = get_log_prob(real_vectors)
                        #log_prob_gen_estimator  = get_log_prob(gen_vectors)
                        #log_prob_real_estimator = get_log_prob(real_vectors)
                        min_cosine, max_cosine, N_STEPS = min(concatted_vectors).item(),max(concatted_vectors).item(), 2000
                        #breakpoint()
                        xs = torch.linspace(min_cosine, max_cosine, N_STEPS).reshape(-1,1) 
                        
                        log_of_prob_multipler = torch.log(torch.tensor([1.0*(max_cosine-min_cosine)/N_STEPS]))
                        log_prob_gen = log_prob_gen_estimator.forward(xs) + log_of_prob_multipler # product becomes sum under log
                        log_prob_real = log_prob_real_estimator.forward(xs) + log_of_prob_multipler # product becomes sum under log

                        prob_gen, prob_real = torch.exp(log_prob_gen), torch.exp(log_prob_real)

                        bhatt_dist = -torch.log(torch.sum(torch.sqrt(prob_gen * prob_real)))
                        loss_tst = bhatt_dist.to('cpu')
                        
                        """bht_coef = torch.tensor([0.0])
                        for x in xs:
                            log_prob_gen = log_prob_gen_estimator.forward(x)
                            log_prob_real = log_prob_real_estimator.forward(x) 
                            #log_prob_gen = log_prob_gen_estimator.log_prob(x)
                            #log_prob_real = log_prob_real_estimator.log_prob(x)
                            prob_gen, prob_real = torch.exp(log_prob_gen), torch.exp(log_prob_real)
                            #breakpoint()
                            bht_coef += torch.sqrt(prob_gen * prob_real) * (max_cosine-min_cosine)/N_STEPS
                        loss_tst = -torch.log(bht_coef).to('cpu')"""
                    
                    elif loss == 'KL':
                        gen_vectors, real_vectors = gen_vectors.to('cpu'), real_vectors.to('cpu')
                        concatted_vectors = torch.cat((gen_vectors, real_vectors), dim=0)
                        min_cosine, max_cosine, N_STEPS = min(concatted_vectors).item(),max(concatted_vectors).item(), 2000
                        xs = torch.linspace(min_cosine, max_cosine, N_STEPS).reshape(-1,1) 

                        log_of_prob_multipler = torch.log(torch.tensor([1.0*(max_cosine-min_cosine)/N_STEPS]))
                        log_prob_gen  = get_log_prob(gen_vectors).forward(xs) + log_of_prob_multipler # product becomes sum under log
                        log_prob_real = get_log_prob(real_vectors).forward(xs) + log_of_prob_multipler # product becomes sum under log
                        
                        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
                        loss_tst = kl_loss(log_prob_gen, log_prob_real).to('cpu')
                    
                    elif loss == 'KS':
                        real_cosines, mixed_cosines = real_vectors.squeeze().to('cpu').numpy(), gen_vectors.squeeze().to('cpu').numpy()
                        ks_stat, _ = ks_2samp(real_cosines, mixed_cosines)
                        loss_tst = ks_stat
                else:
                    if loss == 'FR':
                        loss_tst_fn = tst.statistics_diff.SmoothFRStatistic(batch_size, batch_size, cuda=cuda, compute_t_stat=True)
                    
                    elif loss == 'kNN':
                        loss_tst_fn = tst.statistics_diff.SmoothKNNStatistic(batch_size, batch_size, cuda, k, compute_t_stat=True)
                    
                    else:
                        raise NotImplementedError("message")

                
                if loss=='FR' or loss=='kNN':
                    loss_tst = loss_tst_fn(real_vectors, gen_vectors, alphas =  alphas).to('cpu')

                
                loss_values_for_a_batch_size.append(loss_tst.item())
    
        loss_values.append(np.array(loss_values_for_a_batch_size, dtype=float)) # list of 1D arrays
    
    # call violin plot
    # Save as NPZ
    #arrays_dict = {f'loss_{b_list[i]}': array for i, array in enumerate(loss_values)}
    arrays_dict = {f'loss_{b_list[i]}_{num_cosines_list[i]}_{M_list[i]}': array for i, array in enumerate(loss_values)}
    if use_pca_cosine_logic:
        if loss == 'FR' or loss == 'KL' or loss == 'KS' or loss == 'Bhatt' or loss == 'Bhatt2000':
            plot_loss_violins(loss_values, 
                              b_list, 
                              num_cosines_list,
                              M_list,
                              loss, 
                              save_path=f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_increasing/{loss}_loss_{b_list[0]}_additional_data_' + dir) # number_of_cosines_{num_cosines}/
            np.savez(f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_increasing/{loss}_loss_{b_list[0]}_additional_data_b_list_num_cosine_M_list_' + dir +'.npz', 
                     **arrays_dict)
        
        elif loss == 'kNN':
            plot_loss_violins(loss_values, 
                              b_list, 
                              num_cosines_list,
                              M_list,
                              loss, 
                              k, 
                              save_path=f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_adaptive/k{k}NN_loss_data_' + dir) # number_of_cosines_{num_cosines}/
            np.savez(f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_adaptive/k{k}NN_loss_data_' + dir +'.npz', 
                     **arrays_dict)
    
    else: # use_pca_cosine_logic is False
        if loss == 'FR':
            plot_loss_violins(loss_values, 
                              b_list, 
                              num_cosines_list, 
                              M_list, 
                              loss, 
                              save_path='/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_0/FR_loss_NO_COSINE_data_' + dir)
            np.savez('/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_0/FR_loss_NO_COSINE_data_' + dir +'.npz', 
                     **arrays_dict)
        
        elif loss == 'kNN':
            plot_loss_violins(loss_values, 
                              b_list, 
                              num_cosines_list, 
                              M_list, 
                              loss, 
                              k, 
                              save_path=f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_0/k{k}NN_loss_NO_COSINE_data_' + dir)
            np.savez(f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_0/k{k}NN_loss_NO_COSINE_data_' + dir +'.npz', 
                     **arrays_dict)
