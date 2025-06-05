from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
FIG_DIR = ROOT_DIR / 'analysis/_figs'
DATA_DIR = ROOT_DIR / 'analysis/_data'

ZSL_DIR_SCRATCH = Path('/network/scratch/m/mirceara/zsl_scratch')
ZSL_DIR_OUT= ZSL_DIR_SCRATCH / 'out'
ZSL_DIR_OUT_OLMO = ZSL_DIR_OUT / 'pretraining/olmo'
ZSL_DIR_OUT_NANO = ZSL_DIR_OUT / 'pretraining/nanogpt'
ZSL_DIR_DATA = ZSL_DIR_SCRATCH / 'data'
ZSL_DIR_DATA_NANO = ZSL_DIR_DATA / 'nanogpt'
ZSL_DIR_DATA_OLMO = ZSL_DIR_DATA / 'olmo'
ZSL_DIR_ANALYSIS = ZSL_DIR_SCRATCH / "analysis"