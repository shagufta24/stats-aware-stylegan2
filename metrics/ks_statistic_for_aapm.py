import numpy as np
#import scipy.linalg
#from . import metric_utils

from training.loss import StyleGAN2Loss
from scipy.stats import ks_2samp
import dnnlib
import torch
import copy

#----------------------------------------------------------------------------
def get_reals(opts, max_real, data_loader_kwargs=None):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

     # Initialize.
    num_items = len(dataset)
    if max_real is not None:
        num_items = min(num_items, max_real)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=0, rel_hi=0)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    real_data = []
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=64, **data_loader_kwargs):
        """if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)"""
        #stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        #progress.update(stats.num_items)
        real_data.append(images.detach())

    real_data = torch.cat(real_data)
    real_data = (real_data - 127.5) / 127.5
    return real_data

def get_gens(opts, num_gen, batch_size=64, batch_gen=None, jit=False):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation func.
    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        #img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = (img).clamp(-1, 1)
        return img
    
    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    #########stats = FeatureStats(**stats_kwargs)
    ###########assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=num_gen, rel_lo=0, rel_hi=1)
    ###############detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    gen_data = []

    # Main loop.
    #images = []
    for _ in range(num_gen//batch_size): #while not stats.is_full():
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device, requires_grad=False)
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            gen_data.append(run_generator(z, c).detach())
        #images = torch.cat(images)
        #if images.shape[1] == 1:
            #images = images.repeat([1, 3, 1, 1])
        #features = detector(images, **detector_kwargs)
        #stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        #progress.update(stats.num_items)
        #gen_data.append(images)
    
    gen_data = torch.cat(gen_data)
    return gen_data

def compute_ks_stat(opts, max_real, num_gen):

    sg2loss = StyleGAN2Loss(device=None, G_mapping=None, G_synthesis=None, D=None)
    # get real data
    real_data = get_reals(opts, max_real)

    # get gen images
    gen_data = get_gens(opts, num_gen)
    
    # extract features
    real_vectors = sg2loss.public_features(real_data) # data is tensors of images
    gen_vectors = sg2loss.public_features(gen_data) # data is tensors of images
    real_cosines, mixed_cosines = sg2loss.get_cosines(real_vectors.to('cuda'), gen_vectors.to('cuda'), num_cosines=100000)
    
    if opts.rank != 0:
        return float('nan')
    
    # compute ks_stat
    ks_stat, _ = ks_2samp(real_cosines.squeeze().detach().to('cpu').numpy(), mixed_cosines.squeeze().detach().to('cpu').numpy())  # This will slightly vary over multiple runs, probably in the second decimal place.
    
    return float(ks_stat)

#----------------------------------------------------------------------------
