# Extended: A Systematic Benchmarking Analysis of Transfer Learning for Medical Image Analysis

## Extended work
This is our extension of the [Benchmark repository](https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning) by Hosseinzadeh et al., as we used it in our [Paper]().
Please cite both if you find it helpful. We thank the original authors!
It mainly extends the original repository in the following ways:
- Extended dataset support
- Extended model support and combined repository with (Transformer extension)[https://github.com/jlianglab/BenchmarkTransformers]
- Extended parameterisation
- Extended device support
- Our parameter settings
- Updated requirements.txt file
- New evaluation pipeline: `pipeline_eval.ipynb`

### Getting started
- You can just use the repository as decribed in the original README below
- For Moco-v3 pre-training, use [Moco-v3 extension](https://github.com/felixkrones/moco-v3_f)
  - You need to use deit-converted checkpoints (use convert_to_deit.py for that)
  - For resnet, further prepare the resnet moco checkpoints first using `prep_moco.py` (provide the paths in the file)

### Getting the additional data
- General tips
  - Unzip files
    - `unzip images.zip`
    - `find . -name '*.tar.gz' -exec tar -xf '{}' \;`
  - Deleting files
    - `find . -name '*.tar.gz' -exec rm '{}' \;`
    - `rm images/batch_download_zips.py`
  - Think about where to save files and create folders
    - `mkdir data/raw/name && cd "$_"`
- NIH ChestXray 14:
  - Download data from [box](https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - Download the `images/` folder (there is a nice Python script provided)
  - Download the metadata file `Data_Entry_2017_v2020.csv` into the same folder where the `images/` folder will be
- ChestXpert
    1. Download data from [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
        - Either by directly downloading the zip file 
        - Or by using [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10):
            - Install AzCopy `sudo bash -c "cd /usr/local/bin; curl -L https://aka.ms/downloadazcopy-v10-linux | tar --strip-components=1 --exclude=*.txt -xzvf -; chmod +x azcopy"`
            - Get [Link](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
            - Download: `azcopy copy "LINK" "." --recursive=true`
    2. Create/Copy split file into this folder
        - Either create own file
        - Or use file from [Glocker et al.](https://github.com/biomedia-mira/chexploration/tree/main/datafiles/chexpert)
    3. Unzip all files
        - `cd chexpertchestxrays-u20210408 && unzip CheXpert-v1.0.zip`
- Padchest
  - Download the data from [Padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)
  - Not all, maybe only the 0.zip file. This gives you 1861 manually labelled, frontal view images
    1. Download the [metadata file](https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL/download?path=%2F&files=PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz)
    2. Download the 0.zip folder from (https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL)
- VinDr-CXR
  - Download data from [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)
    1. Only get the test data: `wget -r -N -c -np --user felixkrones --ask-password https://physionet.org/files/vindr-cxr/1.0.0/test/`
    2. Get the annotations: `wget -r -N -c -np --user felixkrones --ask-password https://physionet.org/files/vindr-cxr/1.0.0/annotations/`
    3. Unzip
- OCT
  1. Download the (dataset)[https://data.mendeley.com/datasets/rscbjbr9sj/3]


### Running code in parallel
Run from terminal `torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE ...`

### Running code in background using tmux
1. SSH connect
2. `tmux`
3. Detach: `tmux detach` or `Ctrl+b then d`
4. List sessions: `tmux list-sessions`
4. Resume: `tmux attach -t session_number`


## License

Released under the [ASU GitHub Project License](./LICENSE).


