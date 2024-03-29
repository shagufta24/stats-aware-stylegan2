#conda activate eval_env

path1=/home/nkamath5/stats_aware_gans/aapm_generated_images/new
# outpf='public_metric_113.csv'

for sub_dir in $path1/*/
do
    echo "Sub dir: $sub_dir"
    #python KS_stat_metric.py --num_images 8500 --path_to_reals    enge/challenge_data/train_128 --path_to_fakes $sub_dir --to_save_cosine_plots 0 --convert_size 128 --results_dir metric_results_$subdir --n_trials 5
    python KS_stat_metric.py --num_images 8500 --path_to_reals /shared/rsaas/nkamath5/train_128 --path_to_fakes $sub_dir --to_save_cosine_plots 1 --convert_size 128 --results_dir /home/nkamath5/stats_aware_gans/aapm_eval/ --n_trials 5 --to_save_kde_plots 1
    mv $sub_dir /home/nkamath5/stats_aware_gans/aapm_generated_images/
done

echo "End of eval"

# Author:Nidhish K, Usage: Evaluation of generated images (KS stat) 5 times and storing results in a csv file.
