import sys

sys.path.insert(0, "../")

from deepclustering2.configparser import ConfigManger
from deepclustering2.dataset import ACDCSemiInterface
from deepclustering2.models import Model
from deepclustering2.schedulers import Weight_RampScheduler
from deepclustering2.utils import fix_all_seed

from dataset.augment import val_transform, train_transform
from trainers.Trainer import Trainer

config = ConfigManger("config/config.yaml").config
fix_all_seed(config['seed'])

model = Model(config["Arch"], config["Optim"], config["Scheduler"])
if config['Dataset'] == 'acdc':
    dataset_handler = ACDCSemiInterface(**config["Data"])


def get_group_set(dataloader):
    return set(sorted(dataloader.dataset.get_group_list()))


dataset_handler.compile_dataloader_params(**config["DataLoader"])
label_loader, unlab_loader, val_loader = dataset_handler.SemiSupervisedDataLoaders(
    labeled_transform=train_transform,
    unlabeled_transform=train_transform,
    val_transform=val_transform,
    group_val=True,
    use_infinite_sampler=True,
)
assert get_group_set(label_loader) & get_group_set(unlab_loader) == set()
assert (get_group_set(label_loader) | get_group_set(unlab_loader)) & get_group_set(val_loader) == set()
print(
    f"Labeled loader with {len(get_group_set(label_loader))} groups: \n {', '.join(sorted(get_group_set(label_loader))[:5])}"
)
print(
    f"Unabeled loader with {len(get_group_set(unlab_loader))} groups: \n {', '.join(sorted(get_group_set(unlab_loader))[:5])}"
)
print(
    f"Val loader with {len(get_group_set(val_loader))} groups: \n {', '.join(sorted(get_group_set(val_loader))[:5])}"
)

RegScheduler = Weight_RampScheduler(**config["RegScheduler"])
RegScheduler1 = Weight_RampScheduler(**config["RegScheduler1"])
trainer = Trainer(
    model=model,
    lab_loader=label_loader,
    unlab_loader=unlab_loader,
    weight_scheduler=RegScheduler,
    weight_scheduler1=RegScheduler1,
    val_loader=val_loader,
    config=config,
    **config["Trainer"],
)

trainer.StartTraining()
