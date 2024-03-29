"""Date    : Nov 27, 2023

 Usage : python KS_stat_metric.py --num_images 5000 --path_to_reals /shared/rsaas/nkamath5/challenge_data --path_to_fakes /home/nkamath5/stats_aware_gans/aapm_generated_images/00018-paper512-gamma100-batch48-kimg000907 --to_save_cosine_plots 1 --convert_size 512 --results_dir /home/nkamath5/stats_aware_gans/aapm_eval/00018-paper512-gamma100-batch48-kimg000907 --n_trials 1 --to_save_kde_plots 1
     python KS_stat_metric.py --num_images 5000 --path_to_reals /home/nkamath/AAPM_DGM_Challenge/challenge_data/train_128_temp/train_128 --path_to_fakes /home/nkamath/AAPM_DGM_Challenge/fakes/out_128_training_000115_grouped/out_128_training_000115_snapshot006000 --to_save_cosine_plots 1 --convert_size 128 --results_dir /home/nkamath/AAPM_DGM_Challenge/Eval_AAPM/00115snap06000 --n_trials 1 --to_save_kde_plots 1
     python KS_stat_metric.py --num_images 8500 --path_to_reals /home/nkamath/AAPM_DGM_Challenge/challenge_data/train --path_to_fakes /home/nkamath/AAPM_DGM_Challenge/fakes/out_512_training_000150_grouped/out_512_training_000150_snapshot000600 --to_save_cosine_plots 1 --convert_size 512 --results_dir /home/nkamath/AAPM_DGM_Challenge/Eval_AAPM/00150snap00600 --n_trials 1 --to_save_kde_plots 1
"""

import argparse
import numpy as np
from scipy import stats
import glob
import os, sys
import os.path as p
import imageio as io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib
import seaborn as sns
import cv2
import csv
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

pd.set_option('use_inf_as_na', True)  # All infs throughout are set to nans

def blockPrint():  # from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def load_data(path, num_images, convert_size=512, convert=False):
    fnames = sorted(glob.glob(p.join(path, '*.png')))
    if len(fnames) <= num_images:
        raise ValueError(f"Number of PNG files in either real or generated images dir is less than {num_images}.")
    
    data = np.zeros((num_images, convert_size, convert_size), dtype=np.uint8)

    fnames = fnames[:num_images]
    for i, f in enumerate(fnames):
        print(f"Loading data {i}/{num_images}", end='\r')
        if convert:
            print("Caution: converting real/fake images to convert size. Use same resolutions when possible")
            img = Image.open(f)
            resized = img.resize((convert_size, convert_size), Image.LANCZOS)
            # img = cv2.imread(f, cv2.IMREAD_UNCHANGED) -- cv2 doesn't do anti-aliasing during downsampling
            # resized = cv2.resize(img, (convert_size, convert_size), interpolation=cv2.INTER_AREA) --cv2 no antialiasing
            data[i] = resized
            pass
        else:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if (len(img.shape)==3): # converting rgb to b/w
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data[i] = img
    data = data.astype(int)
    return fnames, data


def public_features(data):
    if (np.all(data==0)):
        print("All data is zero")
        raise ValueError("All data is zero")

    phantom_masks = (data > 45).astype(float)

    print("Computing slice areas")
    areas = np.sum(phantom_masks, axis=(1, 2))

    print("Computing fat area")
    fat = (data >= 45) * (data < 120)
    fat_areas = np.sum(fat, axis=(1, 2))

    print("Computing glandular area")
    gln = (data >= 120) * (data < 226)
    gln_areas = np.sum(gln, axis=(1, 2))

    """#####
    zero_indices = np.where(gln_areas == 0)
    print("Zero indices in gln_areas: ", zero_indices, np.size(zero_indices))
    nan_indices = np.where(np.isnan(gln_areas))
    print("NaN indices in gln_areas: ", nan_indices)
    #####"""

    print("Computing fat to glandular ratio")
    fg_ratios = np.log10(fat_areas / gln_areas)

    N = len(data)
    data = data.reshape(N, -1)

    print("Computing means ...")
    means = np.mean(data, axis=1)

    print("Computing stds ...")
    stds = np.std(data, axis=1)

    print("Computing skewnesses ...")
    skewnesses = stats.skew(data, axis=1)

    print("Computing kurtoses ...")
    kurtoses = stats.kurtosis(data, axis=1)

    print("Computing balances")
    balances = (np.quantile(data, 0.7, axis=1) - np.mean(data, axis=1)) \
               / (np.mean(data, axis=1) - np.quantile(data, 0.3, axis=1))

    return {'mean_pixel_per_img': means,
            'std_of_all_pixels': stds,
            'skewness_per_img': skewnesses,
            'kurtosis_per_img': kurtoses,
            'balance_per_img': balances,

            'area_per_img': areas,
            'fat_area_per_img': fat_areas,
            'gln_area_per_img': gln_areas,
            'fg_ratio_per_img': fg_ratios,
            }


parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("--results_dir", type=str, default='metric_results', help="Results directory")

# data arguments
parser.add_argument("--data_name", type=str, default='downsized_victre_xray_objects2',
                    help="Data name (innermost folder name used for stylegan training)")
parser.add_argument("--path_to_reals", type=str, default='',
                    help="Path to the folder containing the .png images (real or fake)")
parser.add_argument("--path_to_fakes", type=str, default='',
                    help="Path to the folder containing the .png images (real or fake)")
parser.add_argument("--num_images", type=int, default=1000, help="Number of images to load")
parser.add_argument("--levels", type=int, default=64, help='Number of levels used for image digitization')
parser.add_argument("--convert_size", type=int, default=256, help='Resolution (size) of image')
parser.add_argument("--output_filename", type=str, default='public_metric.csv', help='O/p file name to store the KS stats')
parser.add_argument("--n_trials", type=int, default=1, help='No. of times KS stat needs to be computed for same set of fakes')

# summary arguments
parser.add_argument("--num_random_runs", default=100000, type=int,
                    help='Number of samples to draw from while measuring the cosine distance.')
parser.add_argument("--pca_components", default=9, type=int, help='Number of PCA components to use.')
parser.add_argument("--to_save_pca_plots", default=1, type=lambda b: bool(int(b)), help="save pca plots")
parser.add_argument("--to_save_cosine_plots", default=1, type=lambda b: bool(int(b)),
                    help="save cosine plots. Requires seaborn installed.")
parser.add_argument("--to_save_kde_plots", default=1, type=lambda b: bool(int(b)),
                    help="save KDE plots for all domain relevant features. Requires seaborn installed.")
args = parser.parse_args()

if args.results_dir != '':
    #args.results_dir = args.results_dir +"_"+ str(args.path_to_fakes)[-21:-1] +"_" + str(args.num_images)
    args.results_dir = args.results_dir + os.path.basename(os.path.normpath(args.path_to_fakes))  + "_" + str(args.num_images)
    print("Results dir: ", args.results_dir)
    os.makedirs(args.results_dir, exist_ok=True)

# load real or fake data

fake_data = {}
fnames, imgs = load_data(args.path_to_fakes, args.num_images, convert_size=args.convert_size, convert=False) # no conversion needed right now
fake_data['fnames'] = fnames
fake_data['data'] = imgs
print("Fake data loaded")  # we don't load fake data everytime because it stays the same (only 8500 geenrated)

for itr in range(args.n_trials):
    print("Run #", itr+1)
    real_data = {}
    fnames, imgs = load_data(args.path_to_reals, args.num_images, convert_size=args.convert_size, convert=False)  # Changed to False for 128x128 trials
    real_data['fnames'] = fnames
    real_data['data'] = imgs
    print("Real data loaded")

    # Evaluate the public metrics
    blockPrint()
    real_features = public_features(real_data['data'])
    fake_features = public_features(fake_data['data'])

    real_df = pd.DataFrame.from_dict(real_features).fillna(0)
    fake_df = pd.DataFrame.from_dict(fake_features).fillna(0)

    print("Computing real PCA")
    # PCA the reals and store the PCA'd values in a dataframe
    scalar = StandardScaler()
    real_np = real_df.to_numpy()
    scalar.fit(real_np)
    R_scaled = scalar.transform(real_np)
    pca = PCA(n_components=args.pca_components, random_state=1000)
    R_pca = pca.fit_transform(R_scaled)
    dfr_pca = pd.DataFrame(R_pca)

    print("Projecting fake data onto the real PC components")
    fake_np = fake_df.to_numpy()
    F_scaled = scalar.transform(fake_np)
    F_pca = pca.transform(F_scaled)
    dff_pca = pd.DataFrame(F_pca)

    # Compute pairwise cosine distances for the reals
    print("Computing pairwise cosine distances for the reals")
    real_cosines = []
    for i in range(args.num_random_runs):
        print(f'{i}/{args.num_random_runs}', end='\r')

        sampled = dfr_pca.sample(n=2, replace=False).to_numpy()
        real_cosines.append(
            np.dot(sampled[0, :], sampled[1, :]) / (np.linalg.norm(sampled[0, :]) * np.linalg.norm(sampled[1, :])))

    # Compute pairwise cosine distances for real-fake point pairs
    print("Computing pairwise cosine distances for real-fake point pairs")
    mixed_cosines = []
    for i in np.arange(args.num_random_runs):
        sampledr = dfr_pca.sample(n=1, replace=False).to_numpy()
        sampledf = dff_pca.sample(n=1, replace=False).to_numpy()
        mixed_cosines.append(
            np.dot(sampledr[0, :], sampledf[0, :]) / (np.linalg.norm(sampledr[0, :]) * np.linalg.norm(sampledf[0, :])))

    # Compute KS statistic on the reals and the mixed distributions of cosine similarities.
    print("Compute KS statistic")
    ks_stat, _ = ks_2samp(real_cosines,
                          mixed_cosines)  # This will slightly vary over multiple runs, probably in the second decimal place.
    enablePrint()
    print("KS statistic : ", ks_stat)
    ks_stat_str = f'{ks_stat:.6f}'

    if args.to_save_pca_plots:
        print("Saving PCA plots")
        plt.figure()
        plt.scatter(F_pca[:1000, 0], F_pca[:1000, 1], color='red', alpha=0.2, label='fake')
        plt.scatter(R_pca[:1000, 0], R_pca[:1000, 1], color='green', alpha=0.2, label='real')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('PCA')
        plt.legend()
        plt.savefig(p.join(args.results_dir, f'crop_20pca_{itr+1}.png'))
        plt.close()

    if args.to_save_cosine_plots:
        print("Saving cosine plots")
        plt.figure()
        sns.kdeplot(np.array(real_cosines), label='real')
        sns.kdeplot(np.array(mixed_cosines), label='mixed')
        plt.ylabel('density')
        plt.xlabel('cosine similarity')
        plt.title(f'KS statistic {args.num_images} : {ks_stat_str}')
        plt.legend()
        plt.savefig(p.join(args.results_dir, f'10cosine_{itr+1}.png'))
        plt.close()
    
    if args.to_save_kde_plots and (itr == 1):
        print("Saving feature wise KDE plots")
        common_keys = real_features.keys()
        for key in common_keys:
            real_feature = real_features[key]  # Numpy array from real_features
            fake_feature = fake_features[key]  # Numpy array from fake_features
            
            # Create KDE plots for each feature from real and fake data
            plt.figure()#figsize=(10, 6))
            #sns.set(style="whitegrid")
            sns.kdeplot(real_feature, label=f"Real {key}", fill=True)
            sns.kdeplot(fake_feature, label=f"Fake {key}", fill=True)
            plt.xlabel(f"{key}")
            plt.ylabel("Density")
            plt.title(f"KDE Plot for {key} Real vs. Fake")
            plt.legend()
            plt.savefig(p.join(args.results_dir, f'{key}_KDE_plots_{itr+1}.png'))
            plt.close()

    with open(p.join(args.results_dir, f'public_metric_{itr+1}.txt'), 'w') as fid:
        fid.write(f"KS statistic   : {ks_stat:.6f}")

    if args.n_trials > 1:
        output_filename = args.output_filename
        if not os.path.exists(output_filename):
            with open(output_filename, 'w', newline='') as csvfile:
                fieldnames = ['Metric', 'Fakes network snap', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'Metric': 'KS statistic', 'Fakes network snap': args.path_to_fakes[-31:-1], 'Value': f'{ks_stat:.6f}'})
        else:
            with open(output_filename, 'a', newline='') as csvfile:
                fieldnames = ['Metric', 'Fakes network snap', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Metric': 'KS statistic', 'Fakes network snap': args.path_to_fakes[-31:-1], 'Value': f'{ks_stat:.6f}'})
    if itr == 1:  # plots will be the same regardless of run/trial number
        common_keys = real_features.keys()
        real_keys = ["real_"+key for key in common_keys]
        fake_keys = ["fake_"+key for key in common_keys]
        real_features = dict(zip(real_keys, list(real_features.values())))
        fake_features = dict(zip(fake_keys, list(fake_features.values())))
        np.savez_compressed(p.join(args.results_dir, 'numpy_arrays_for_plots.npz'), F_pca=F_pca[:1000,:], R_pca=R_pca[:1000,:], real_cosines=real_cosines, mixed_cosines=mixed_cosines, ks_stat=np.array(ks_stat), **real_features, **fake_features)

