# Download OLMo-7B-0724 pretraining data locally

Use [aria2](https://aria2.github.io/) (also available as python [wheel](https://github.com/WSH032/aria2-wheel/) on pip) to efficiently download ~3TiB of preprocessed pretraining data.

`0724-urls.txt` contains data paths from the official OLMo-7B-0724 [config](https://github.com/allenai/OLMo/blob/8589c38756ac3359abbe5938dfbeaff20e92c3a1/configs/official/OLMo-7B-0724.yaml), publically available at the time of writing (October 2024). `create-aria2c-input.py` converts these to an [input file](https://aria2.github.io/manual/en/html/aria2c.html#id2) `0724-urls-aria2.txt` for aria2c, preserving the directory structure relative to the working directory.

The dataset can then be downloaded with e.g.:

```bash
cd $ZSL_OLMO_DAT_DIR
aria2c -x 8 -s 8 --max-concurrent-downloads=8 \ 
    -i $ZSL_OLMO_SRC_DIR/zsl/utils/download_data/0724-urls-aria2.txt \ 
    --dir=./ \ 
    --continue=true 
```