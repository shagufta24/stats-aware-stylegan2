# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import sys
#sys.path.append('/home/nkamath5/miniconda3/envs/debug_tst_env/lib/python3.6/site-packages')
##sys.path.append('/home/nkamath5/miniconda3/pkgs/torch-two-sample-0.1-py36h39e3cac_0/lib/python3.6/site-packages')
#print(sys.path)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_to_tst = os.path.join(current_dir, 'torch-two-sample')
sys.path.append(relative_path_to_tst)
import torch_two_sample as tst
##import torch_two_sample as tst  # edited
import torch.distributed as dist
import torch.nn.functional as F

from pytorch_generative.models import kde
#import pyro  # we will not use pyro until we have read pyro documentation properly
#import pyro.distributions as pyro_dist

import warnings  # edited
warnings.filterwarnings("ignore")  # edited
#breakpoint()

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, 
                 alphas=[0.1], tst_pl_wts_ratio=0, features=['all'], num_random_runs=100, loss_name=None):  
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        #---------edit start-----
        self.tst_weight = tst_pl_wts_ratio * pl_weight
        self.debug_flag = True
        self.alphas = alphas
        self.features = features
        self.num_random_runs = num_random_runs  # number of cosines
        self.loss_name = loss_name
        # loss_name is supposed to be None only when we use functions from this class for 
        # evaluation of our custom metrics
        if not loss_name is None:
            print("num_random_runs =", self.num_random_runs)
            print("Weight of tst_loss = ", self.tst_weight)
            print("List of alphas used", self.alphas)
            print(f"Loss name: {loss_name}")
        #breakpoint()
        #---------edit end-----

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

#------Start of edit------

    def public_features(self, data):

        final_features = [] # keep appending relevant features to this list & convert to tensor at the end

        # Medical features

        all_medical = False
        if 'all_medical' in self.features:
            all_medical = True
        data          = torch.squeeze(data).float()  # data is already expected to be in float, but leaving the .float() here
        if all_medical or ('areas' in self.features):
            phantom_masks = (data > -0.6471).float()  # [(45/127.5)-1] = -0.6471
            # gradients are zero at all places where a step function is differentiable, hence error signal is not passed to gen images via this route
            areas         = torch.sum(phantom_masks, dim=(1,2)) #.unsqueeze(1)
            final_features.append(areas)
        if all_medical or ('fg_ratios' in self.features) or ('fat_areas' in self.features):
            fat           = (data >= -0.6471) * (data < -0.0588)  # [(120/127.5)-1] = -0.0588
            fat_areas     = torch.sum(fat, dim=(1,2)) #.unsqueeze(1)
            if all_medical or ('fat_areas' in self.features): final_features.append(fat_areas)
        if all_medical or ('fg_ratios' in self.features) or ('gln_areas' in self.features):
            gln           = (data >= -0.0588) * (data < 0.7725)  # [(226/127.5)-1] = 0.7725
            gln_areas     = torch.sum(gln, dim=(1,2)) #.unsqueeze(1)
            if all_medical or ('gln_areas' in self.features): final_features.append(gln_areas)
        if all_medical or ('fg_ratios' in self.features):
            fg_ratios     = torch.log10(((fat_areas + 0.00012)/ (gln_areas + 0.0001)) + 1)
            final_features.append(fg_ratios)
            # 1 added since it protects from NaN values for low values inside log, without affecting gradient a lot (check graphically for clarity)
            #print("data ", data.requires_grad, " phantom_masks ", phantom_masks.requires_grad, " areas ", areas.requires_grad)
            #print("fat_areas ", fat_areas.requires_grad, " gln_areas ", gln_areas.requires_grad, " fg_ratios ", fg_ratios.requires_grad)
        N = len(data)  # alternatively N = data.shape[0], i.e. batch size
        data          = data.reshape(N, -1)
        #print("data: ", data.shape)
        if all_medical or 'means' in self.features or 'skewnesses' in self.features or 'kurtoses' in self.features or 'balances' in self.features:
            means         = torch.mean(data, dim=1) #.unsqueeze(1)
            if all_medical or ('means' in self.features): final_features.append(means)
            #print("means: ", means.shape)
        if all_medical or 'stds' in self.features or 'skewnesses' in self.features or 'kurtoses' in self.features:
            stds          = torch.std(data, dim=1) #.unsqueeze(1)
            if all_medical or ('stds' in self.features): final_features.append(stds)
            #print("stds: ", stds.shape)
        if all_medical or 'skewnesses' in self.features:
            #print(f"means.shape: {means.shape}") # ([16])
            #print(f"data.shape: {data.shape}")  # ([16, 262144*])
            #print(f"stds.shape: {stds.shape}")  # ([16])
            #print(f"stds.unqueeze(-1) shape: {(stds.unsqueeze(-1)).shape}")
            skewnesses    = (torch.mean((data - means.unsqueeze(-1))**3, dim=1) / stds.squeeze()**3) #.unsqueeze(1)
            final_features.append(skewnesses)
        if all_medical or 'kurtoses' in self.features:
            kurtoses      = ((torch.mean((data - means.unsqueeze(-1))**4, dim=1) / stds.squeeze()**4) - 3.0) #.unsqueeze(1)
            final_features.append(kurtoses)
            #print(torch.quantile(data, 0.7, dim=1).shape)
        if all_medical or 'balances' in self.features:
            balances = (torch.quantile(data, 0.7, dim=1) - means) / (means - torch.quantile(data, 0.3, dim=1))
                        #( torch.quantile(data, 0.7, dim=1).unsqueeze(1) - means ) \
                        #/ ( means - torch.quantile(data, 0.3, dim=1).unsqueeze(1) )
            final_features.append(balances)

        # Materials features
            
        all_materials = False
        if 'all_materials' in self.features:
            all_materials = True

        data          = torch.squeeze(data).float()  # data is already expected to be in float, but leaving the .float() here
        if all_materials or ('p1_value' in self.features):
            p1 = data.mean()
            final_features.append(torch.sum(p1, dim=(1,2)))

        if all_medical or ('p2_value' in self.features):
            radvars = []
            dimX = data.shape[0]
            dimY = data.shape[1]
            fftimage = np.fft.fft2(data)
            final_image = np.fft.ifft2(fftimage*np.conj(fftimage))
            finImg = np.abs(final_image)/(dimX*dimY)
            centrdImg = np.fft.fftshift(finImg)
            center = [int(dimX/2), int(dimY/2)]

            x, y = np.indices(centrdImg.shape)
            rad = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            ind = np.argsort(rad.flat)
            rad_sorted = rad.flat[ind]
            image_sorted = centrdImg.flat[ind]
            rad_round = rad_sorted.astype(int)
            deltar = rad_round[1:] - rad_round[:-1]
            nonzero_deltar = np.where(deltar > 0.0)[0]
            nind = nonzero_deltar[1:] - nonzero_deltar[:-1]
            yvalues = np.cumsum(image_sorted, dtype = np.float64)
            yvalues = yvalues[nonzero_deltar[1:]] - yvalues[nonzero_deltar[:-1]]
            radial_var = yvalues/nind
            radvars.append(radial_var)
            p2_vec = np.array(radvars)
            p2 = p2_vec[0]
            final_features.append(torch.sum(p2, dim=(1,2)))
                  
        feature_vectors = torch.stack(final_features, dim=1).squeeze(1)
        if not feature_vectors.isfinite().all():
            print("feature_vectors.isfinite().all() is False")
            breakpoint()
        return feature_vectors
        #return stds
        #return torch.stack([stds, balances], dim=1).squeeze()  # since only these have multi(bi)modal distributions like F/G ratio which has 3/4 modes
        ##return torch.stack([means, stds, skewnesses, kurtoses, balances, areas, fat_areas, gln_areas, fg_ratios], dim=1).squeeze()
        #return torch.stack([means, fg_ratios], dim=1).squeeze()
    
    def get_cosines(self, real_vectors, gen_vectors, num_cosines):
        
        # standardize before pca (somehow keeping center=True in pca_lowrank wasn't giving correct results)
        real_vectors = real_vectors - real_vectors.mean(dim=0) # dim to reduce is that of num_samples
        real_vectors = real_vectors / real_vectors.std(dim=0)
        gen_vectors  = gen_vectors  - gen_vectors.mean(dim=0)
        gen_vectors  = gen_vectors  / gen_vectors.std(dim=0)

        _, _, V = torch.pca_lowrank(real_vectors, q=real_vectors.shape[1], center=False) 
        # added the q parameter since by default it would have been 6 & we want it to be overestimated: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        # matmul(A, V[:, :k]) projects data to the first k principal components
        real_vectors = torch.matmul(real_vectors, V[:, :real_vectors.shape[1]]) # pca projections
        gen_vectors = torch.matmul(gen_vectors, V[:, :real_vectors.shape[1]]) # projecting onto p components of reals
        # real_vectors & gen_vectors must now be of shape (batch_size_per_gpu x k)
        
        #sample 2 random vectors from reals & find their cosine similarities
        # create a (2x batch_size) tensor; row 0 are indices for first vectors, row 1 for second in cos sim calculation
        random_matching = torch.randint(low=0, high=real_vectors.shape[0], size=(2, num_cosines))
        real_vecs_set_1 = real_vectors[random_matching[0]]  # size (num_random_runs, k)
        real_vecs_set_2 = real_vectors[random_matching[1]]
        cossim_reals = F.cosine_similarity(real_vecs_set_1, real_vecs_set_2, dim=1, eps=1e-8)

        random_matching = torch.randint(low=0, high=real_vectors.shape[0], size=(2, num_cosines))
        real_vecs_set_1 = real_vectors[random_matching[0]]
        gen_vecs_set_2 = gen_vectors[random_matching[1]]
        cossim_mixed = F.cosine_similarity(real_vecs_set_1, gen_vecs_set_2, dim=1, eps=1e-8)

        # cosine similarity vectors become our final real & gen vectors.
        real_vectors = cossim_reals.unsqueeze(1) # to make 1-D tensor into 2-D tensor for purposes of code in two_sample_test
        gen_vectors = cossim_mixed.unsqueeze(1)
        return real_vectors, gen_vectors
    
    def get_log_prob(self, vectors):
        """ # inputs: vectors must be n x 1
            # returns: estimator fitted to vectors using kde 
            # estimator.forward() will return log probs
            
            We use the scotts_factor as a heuristic to compute bandwidth in accordance with scipy implementation of kde
            
            Currently we have two ways of doing kde:-
                1. Using kde in https://github.com/EugenHotaj/pytorch-generative
                2. Using the pyro library https://pyro.ai/examples/index.html
            Both have a space complexity of O(training_vector_size * test_vector_size)
            where training_vectors are those at which kernels are fitted and 
            testing_vectors are those where prob/log_prob are estimated"""
        
        scotts_factor = self.num_random_runs**(-1.0/(4+vectors.shape[1])) # heuristic for bandwidth computation
        mean_vector   = vectors.mean(dim=0) # mean for each dimension/ col, so taken across rows
        normalized_vectors = vectors - mean_vector.unsqueeze(0)  # nxd - 1xd
        covariance    = torch.mm(torch.t(normalized_vectors), normalized_vectors) / (vectors.shape[0]-1) # (1xn) @ (nx1)
        bandwidth     = covariance*scotts_factor**2 *11 # *11 added because it just seems to work while comparing with scipy implementation
        log_prob_estimator = kde.KernelDensityEstimator(vectors, kde.GaussianKernel(bandwidth.item()))
        ###log_prob_estimator = pyro_dist.Normal(vectors.squeeze(), bandwidth)
        return log_prob_estimator

#------End of edit------

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        #------Start of edit------
        do_tst   = (phase in ['Greg', 'Gboth']) and (self.tst_weight != 0)
        #------End of edit------
        
        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl and not do_tst)) # May get synced by Gpl or Gtst.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=(sync and not do_tst)) # May get synced by Gtst
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                # start of edit --------------
                #if self.debug_flag:
                    #print("loss_Gpl is on device: ", loss_Gpl.get_device())
                # end of edit --------------
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
        
        #------Start of edit------
        
        # tst: Apply Two Sample Test to incentivize similar medical statistics in reals & fakes
        if do_tst:
            with torch.autograd.profiler.record_function('G_two_sample_forward'):
                batch_size = gen_z.shape[0]
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=sync)
                #current_device = gen_img.get_device()
                #print("gen_img.requires_grad should be True: ", gen_img.requires_grad)
                #print("gen_img min & max values: ", torch.min(gen_img), " & ", torch.max(gen_img))
                #gen_vectors = (self.public_features(gen_img)).to(current_device)
                #real_vectors = (self.public_features(real_img.detach().requires_grad_(False).to(current_device))).to(current_device)
                gen_vectors = self.public_features(gen_img)
                real_vectors = self.public_features(real_img)
                if self.num_random_runs > 0: # self.use_pca_logic == True:
                    real_vectors, gen_vectors = self.get_cosines(real_vectors, gen_vectors, num_cosines=self.num_random_runs)
                    if not real_vectors.isfinite().all() or not gen_vectors.isfinite().all():
                        print(f"real_vectors.isfinite().all() and gen_vectors.isfinite().all() is False")
                        breakpoint()
                    real_cosines_in_range = real_vectors.max().item() < 1.000001 and real_vectors.min().item() > -1.000001
                    gen_cosines_in_range  = gen_vectors.max().item() < 1.000001 and gen_vectors.min().item() > -1.000001
                    if not (real_cosines_in_range and gen_cosines_in_range):
                        print(f"Some cosines not in range")
                        #breakpoint()
                    #print(f"{real_vectors.shape}, and  {gen_vectors.shape}")

                #print("real & gen shape", real_vectors.shape, gen_vectors.shape)
                #print("gen_vectors.requires_grad should be True: ", gen_vectors.requires_grad)
                
                # Code for gathering input_data from all processes to process 0
                world_size = torch.cuda.device_count()
                batch_size = batch_size * world_size # new batch will include entries from all GPUs
                if world_size > 1: # distributed training
                    group = dist.new_group([i for i in range(world_size)])
                    # dist.gather(input_data, gather_list=None, dst=0, group=group) not supported by process group nccl https://github.com/pytorch/pytorch/issues/55893#issuecomment-1022727055
                    with torch.no_grad():
                        gen_tensor_list = [torch.empty_like(gen_vectors) for _ in range(world_size)]
                        dist.all_gather(gen_tensor_list, gen_vectors, group)
                        real_tensor_list = [torch.empty_like(real_vectors) for _ in range(world_size)]
                        dist.all_gather(real_tensor_list, real_vectors, group)
                    rank = dist.get_rank()
                    gen_tensor_list[rank] = gen_vectors
                    real_tensor_list[rank] = real_vectors
                    print(".\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.")
                    for t in gen_tensor_list:
                        print(t.device)
                    print(".\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.")
                    all_gen_tensor = torch.cat(gen_tensor_list)
                    all_real_tensor = torch.cat(real_tensor_list)
                    print(f"gen_img.get_device(): {gen_img.get_device()}, all_gen_tensor.get_device(): {all_gen_tensor.get_device()} and rank is {rank}")
                    print(f"real_img.get_device(): {real_img.get_device()}, all_real_tensor.get_device(): {all_real_tensor.get_device()} and rank is {rank}")
                    all_gen_tensor = all_gen_tensor.to(gen_img.get_device())
                    all_real_tensor = all_real_tensor.to(real_img.get_device())
                else:  # 1 GPU, no distributed training
                    all_real_tensor = real_vectors
                    all_gen_tensor = gen_vectors
                
                # set up loss function object for FR/ kNN test based on num_random_runs and loss_name
                # get log_probs via kde for KL div/ Bhattacharya dist
                if self.num_random_runs > 0:
                    if self.loss_name == 'FR':
                        loss_tst_fn = tst.statistics_diff.SmoothFRStatistic(self.num_random_runs, self.num_random_runs, cuda=True, compute_t_stat=True) # since the datapoints being compared for similar distribution are now num_random_runs in number
                    elif self.loss_name == 'kNN':
                        loss_tst_fn = tst.statistics_diff.SmoothKNNStatistic(self.num_random_runs, self.num_random_runs, cuda=True, k=1, compute_t_stat=True) 
                    elif self.loss_name == 'Bhatt' or self.loss_name == 'KL': # Bhattacharya distance or KL divergance
                        # we will use our data points/vectors (i.e. cosines) to estimate PDFs using KDE and 
                        # then use the probability densities to compute the Bhattacharyya distance between the PDF of real & PDF of gen
                        # hint: KDE estimation takes memory as O(size(real_vecs) * size(gen_vecs)) depending on implementation
                        # hint: so it may give memory errors for high num_random_runs / num_cosines
                        
                        #concatted_vectors = torch.cat((all_gen_tensor, all_real_tensor), dim=0)  # assuming all_x_tensor to be 2-D with dim-0 corresponding to num of vectors
                        #min_cosine, max_cosine = min(concatted_vectors).item(),max(concatted_vectors).item()
                        min_cosine, max_cosine = -1.0, 1.0 # -1 & 1 since pdf should be evaluated in the entire valid range, else discrete probabilties in BHatt or KL div formula wouldn't sum = (or close to) 1

                        #real_minmaxdiff = max(all_real_tensor).item() - min(all_real_tensor).item()
                        #gen_minmaxdiff = max(all_gen_tensor).item() - min(all_gen_tensor).item()
                        
                        # N_STEPS was originally meant to be 2000; such that in the case cosines range from [-1,1], the 
                        # increment that we'll see would be 2/2000 = 1e-3 in size.
                        # However, if kde for gen or reals is such that its peak has a footprint <<< 1e-3 and if the kde 
                        # is almost 0 everywhere else, then during the integration/summing during Bhattacharyya loss computation
                        # estimated prob will be effectively very high. In order to handle this, we will choose to seek to break the
                        # whole range of the gen cosines/ real cosines down into 2000 parts, such that the number of steps needed to 
                        # cover the total range [-1, 1] is not above a certain high threshold say 1e8.

                        #step_size = max(min(real_minmaxdiff/2000, gen_minmaxdiff/2000), 2e-8)
                        
                        N_STEPS = 20000 #min(max(4000//real_minmaxdiff, 4000//gen_minmaxdiff), 1e5)# 2000
                        xs = torch.linspace(min_cosine, max_cosine, int(N_STEPS), device=self.device)[:-1].reshape(-1,1) 
                        
                        bin_width = 1.0 * (max_cosine - min_cosine)/N_STEPS
                        log_prob_gen = self.get_log_prob(all_gen_tensor).forward(xs)
                        log_prob_real = self.get_log_prob(all_real_tensor).forward(xs)

                        if not log_prob_real.isfinite().all() or not log_prob_gen.isfinite().all():
                            print("log_prob are not finite")
                            breakpoint()
                        
                        # instead of concatted vectors, we could also have used points from the linear space of min(real_vec, gen_vec)
                        # and max(real_vec, gen_vec) as done in 
                        # https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py
                    else:
                        raise ValueError(f"{self.loss_name} is not an implemented loss")
                else: # no cosines
                    if self.loss_name == 'FR':
                        loss_tst_fn = tst.statistics_diff.SmoothFRStatistic(batch_size, batch_size, cuda=True, compute_t_stat=True)
                    elif self.loss_name == 'kNN':
                        loss_tst_fn = tst.statistics_diff.SmoothKNNStatistic(batch_size, batch_size, cuda=True, k=1, compute_t_stat=True)
                    else:
                        raise ValueError(f"{self.loss_name} is not an implemented without using cosine logic (hint: set num_random_runs or num_cosines > 0)")                

                if self.loss_name == 'FR' or self.loss_name == 'kNN':
                    loss_tst = loss_tst_fn(all_real_tensor, all_gen_tensor, alphas = self.alphas)  # https://torch-two-sample.readthedocs.io/en/latest/#torch_two_sample.statistics_diff.SmoothFRStatistic.__call__
                    # used alphas = [0.1] for 128x128 as the first choice since the same value was used in a demo notebook https://github.com/josipd/torch-two-sample/blob/master/notebooks/mnist.ipynb - this choice seemed to work
                elif self.loss_name == 'Bhatt':
                    prob_gen, prob_real = torch.exp(log_prob_gen), torch.exp(log_prob_real)
                    if not prob_real.isfinite().all() or not prob_gen.isfinite().all():
                        print("prob real or gen are not finite")
                        breakpoint()

                    #real_prob_in_range = prob_real.max().item() < 1.00001 and prob_real.min().item() > -0.00001
                    #gen_prob_in_range = prob_gen.max().item() < 1.00001 and prob_gen.min().item() > -0.00001
                    #if not real_prob_in_range or not gen_prob_in_range:
                        #if not real_prob_in_range:
                            #print("Real prob not in range") #.. applying softmax")
                        #if not gen_prob_in_range:
                            #print("Gen prob not in range") #.. applying softmax")
                        
                        #breakpoint()
                    
                    prob_real = torch.clamp(prob_real, min=0.0, max=1.0) #torch.nn.functional.relu(prob_real)
                    prob_gen = torch.clamp(prob_gen, min=0.0, max=1.0) #torch.nn.functional.relu(prob_gen)
                    
                    loss_tst = - 1.0 * torch.log((torch.sum(torch.sqrt(prob_gen * prob_real)) * bin_width) + 0.01) # 0.01 added for protecting from disjoint supports (3/10/24)
                    if not loss_tst.isfinite().all():
                        print("Loss is not finite")
                        breakpoint()
                elif self.loss_name == 'KL':
                    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
                    loss_tst = kl_loss(log_prob_gen, log_prob_real)
                
                training_stats.report('Loss/loss_two_sample', loss_tst)
                loss_tst = world_size * loss_tst * self.tst_weight  # multiplying by world_size since DDP will average out the gradients, but we want to sum - https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
                training_stats.report('Loss/G/reg_two_sample', loss_tst)
            with torch.autograd.profiler.record_function('G_two_sample_backward'):
                if self.debug_flag:
                    print("List of alphas = ", self.alphas)
                    #print("Device of real_img, gen_img, real_vectors, gen_vectors : ", real_img.get_device(), gen_img.get_device(), real_vectors.get_device(), gen_vectors.get_device())
                    #print("loss_tst is on device: ", loss_tst.get_device())
                    self.debug_flag = False
                loss_tst.mul(gain).backward()

        #------End of edit------

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

