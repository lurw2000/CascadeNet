'''
Modify config json files in ./cascadegan_sigcomm24/*.json 
to search for better hyperparameters.
Save new json in ./cascadegan_sigcomm24/param_search

cd config/param_search && sbatch slurm_4.slurm
'''

import os
import json
import itertools
import collections

# first clear the directory
os.system('rm -rf ./config/param_search/*')
os.system('mkdir ./config/param_search')

hyperparam_lst = list(itertools.product(
    ['ca', 'caida', 'dc' , 'ton', 'test'], # the dataset the original config file corresponds to
    [True, False], # use_feature_extraction
    [True, False], # output_zero_flag
    [50, 200, 1000, 2000], # pkt_rate_sample_method
))

# save a csv
with open('./config/hyperparam_lst.csv', 'w') as f:
    f.write('i,dataset,use_feature_extraction,output_zero_flag,pkt_rate_sample_method\n')
    for i, hyperparams in enumerate(hyperparam_lst):
        f.write(f'{i},{hyperparams[0]},{hyperparams[1]},{hyperparams[2]},{hyperparams[3]}\n')

for i, hyperparams in enumerate(hyperparam_lst):
    (dataset, use_feature_extraction, output_zero_flag, pkt_rate_sample_method) = hyperparams
    description = f'{i}-{dataset}-feature_{use_feature_extraction}-zero_flag_{output_zero_flag}-rate_{pkt_rate_sample_method}'
    
    # dataset
    with open(f'./config/cascadegan_sigcomm24/config-{dataset}-vanilla_LM-5_200.json', 'r') as f:
        config = json.load(f)
    
    # use_feature_extraction
    if not use_feature_extraction:
        config['pre_post_processor']['feature_extraction']['methods'] = {
            "simply_zeros": {
                "use_log": True
            },
            "simply_zeros2": {
                "use_log": True
            },
        }
    # output_zero_flag
    config['pre_post_processor']['output_zero_flag'] = output_zero_flag
    # NOTE ...
    config['gan']['generator']['packetrate']['zero_flag'] = output_zero_flag
    
    # pkt_rate_sample_method
    config['pre_post_processor']['pkt_rate_sample_method'] = ["time_point_count", pkt_rate_sample_method]

    # change num_workers from 8 to 1
    for name in ['flowlevel', 'packetrate', 'packetfield', 'cascade_comp']:
        config['dataloader'][name]['num_workers'] = 1 

    # output to folder named number i (i in 0, 1, ..., 63)
    config['pre_post_processor']['output_folder'] = \
        f'/gpfsnyu/scratch/dx2102/NetworkML/result/param_search/{i}'
    
    # annotate 'hyperparams' in the beginning of the file
    config = collections.OrderedDict(
        [('hyperparams', {
        'dataset': dataset,
        'use_feature_extraction': use_feature_extraction,
        'output_zero_flag': output_zero_flag,
        'pkt_rate_sample_method': pkt_rate_sample_method
        })] 
        + list(config.items())
    )

    path = f'/gpfsnyu/home/dx2102/ML-testing-dev/config/param_search/{description}.json'
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

    slurm = f'''
    #!/bin/bash
    #SBATCH --job-name=NetML-{i}
    #SBATCH --partition=netsys
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1

    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-gpu=15
    #SBATCH --mem-per-gpu=120GB
    #SBATCH --output=/gpfsnyu/scratch/dx2102/NetworkML/logs/{i}.out
    #SBATCH --error=/gpfsnyu/scratch/dx2102/NetworkML/logs/{i}.out

    date
    module load anaconda3
    source activate /gpfsnyu/home/dx2102/.conda/envs/cascadenet
    cd ~/ML-testing-dev/demo/
    path={path}
    time python training.py --path=$path
    time python evaluating.py --path=$path
    time python evaluating_timestamp_ratio.py --path=$path
    time python evaluating-median_span.py --path=$path
    time python evaluating-equal.py --path=$path
    time python generate_copy.py --path=$path
    date
    '''
    slurm = slurm.strip()
    slurm = '\n'.join([line.strip() for line in slurm.split('\n')])
    # with open(f'./config/param_search/{i}.slurm', 'w') as f:
    #     f.write(slurm)
    