for sub_dir in */
do
    #python KS_stat_metric.py --num_images 8500 --path_to_reals /home/nkamath/AAPM_DGM_Challenge/challenge_data/train_128 --path_to_fakes $sub_dir --to_save_cosine_plots 0 --convert_size 128 --results_dir metric_results_$subdir --n_trials 5
    #python KS_stat_metric.py --num_images 8500 --path_to_reals /shared/rsaas/nkamath5/train_128 --path_to_fakes /home/nkamath5/stats_aware_gans/aapm_generated_images/00026-128-auto1-gamma2-batch64-kimg000604 --to_save_cosine_plots 1 --convert_size 128 --results_dir /home/nkamath5/stats_aware_gans/aapm_eval/00026-128-auto1-gamma2-batch64-kimg000604 --n_trials 5 --to_save_kde_plots 1
    echo $sub_dir
    echo metric_results_$sub_dir
done

echo "End of eval"


