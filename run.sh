#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=def-chdesa
save_dir=vat_group

ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(
"python -O main.py Optim.lr=0.00001 Trainer.save_dir=${save_dir}/f_0.00001r Data.unlabeled_data_ratio=0.01 Data.labeled_data_ratio=0.99"
"python -O main.py Optim.lr=0.00001 Trainer.save_dir=${save_dir}/p_0.00001r Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"


#"python -O main.py Optim.lr=0.00001 Train_vat=True RegScheduler.max_value=0.05 Trainer.save_dir=${save_dir}/vat_0.05 Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Optim.lr=0.00001 Train_vat=True RegScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/vat_0.1 Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Optim.lr=0.00001 Train_vat=True RegScheduler.max_value=0.5 Trainer.save_dir=${save_dir}/vat_0.5 Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
