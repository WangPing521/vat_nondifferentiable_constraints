#!/usr/bin/env bash

save_dir=vat_group


ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(
"python -O main.py Optim.lr=0.00001 Train_vat=True Trainer.save_dir=${save_dir}/vat1_0 Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)
gpuqueue "${StringArray[@]}" --available_gpus 2 4