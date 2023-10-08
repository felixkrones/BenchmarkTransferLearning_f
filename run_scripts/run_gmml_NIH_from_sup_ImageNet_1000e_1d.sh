#! /bin/bash 
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint='gpu_mem:16GB'
#SBATCH --time=47:10:00
#SBATCH --partition=long
#SBATCH --job-name=GMML_sup_1d
#SBATCH --clusters=htc

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/gmmlenv
conda info --env

data_set=ChestXray14
data_dir=/data/inet-multimodal-ai/wolf6245/data/NIH/images
model=vit_small
mode=train
resume=True
init=GMML
proxy_dir=/data/inet-multimodal-ai/wolf6245/src/medical_gmml-main_forked/checkpoints/vit_small/Pretrain/CXR8/timm_1D/checkpoint1000.pth
train_list=dataset/Xray14_train_official.txt
val_list=dataset/Xray14_val_official.txt
test_list=dataset/Xray14_test_official.txt
device=cuda
epochs=30
batch_size=64
trial=5
nc=1

python  main_classification.py --data_set $data_set --data_dir $data_dir --model $model --mode $mode --resume $resume --init $init --proxy_dir $proxy_dir --train_list $train_list --val_list $val_list --test_list $test_list --device $device --epochs $epochs --batch_size $batch_size --trial $trial --nc $nc
