from pathlib import Path
import yaml

EXP_DIR = Path(__file__).parent
BASE_CONFIGS = list(EXP_DIR.glob('*-base.yaml'))

max_duration = int(2**15)
seq_lens = [1024,2048]
tokens_bszs = [int(2**20), int(2**22)] # 1M, 4M
gridsearch = [(seq_len, token_bsz) for seq_len in seq_lens for token_bsz in tokens_bszs]
model_to_seq_to_bsz = {
    'olmo_14M': {1024: 64, 2048: 32},
    'olmo_144M': {1024: 32, 2048: 32},
}

# load, edit, and save yaml
for base_config in BASE_CONFIGS:
    model = base_config.stem.split('-')[0]
    for seq_len, token_bsz in gridsearch:
        config = yaml.safe_load(base_config.read_text())
        config['max_duration'] = max_duration
        config['model']['max_sequence_length'] = seq_len
        config['global_train_batch_size'] = token_bsz // seq_len
        config['device_train_microbatch_size'] = model_to_seq_to_bsz[model][seq_len]
        token_bsz_str = f"{int(token_bsz // 1e6)}M"
        exp_id = f"{model}-bsz_{token_bsz_str}-seq_{seq_len}"
        job_id = f"{exp_id}"
        f = f"{exp_id}.yaml"
        print(f"Writing {f}")
        with open(EXP_DIR / f, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        # base sbatch
        base_sbatch_path = EXP_DIR / f'{model}-base.sbatch'
        sbatch = base_sbatch_path.read_text().splitlines()
        sbatch[1] = f'#SBATCH --job-name={job_id}'
        assert sbatch[24].startswith('export OLMO_TRAIN_CFG=')
        sbatch[24] = f'export OLMO_TRAIN_CFG=$ZSL_DIR/pretraining/configs/ablation-bsz_seq/{exp_id}.yaml'
        # write sbatch
        sbatch_path = EXP_DIR / f'{exp_id}.sbatch'
        print(f"Writing {sbatch_path}")
        with open(sbatch_path, 'w') as f:
            f.write('\n'.join(sbatch))

