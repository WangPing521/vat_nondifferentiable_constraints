
from typing import Union, Dict, Any

from deepclustering2.meters2 import MeterInterface

from loss_tool.Vatloss import VATLoss
from loss_tool.constraint_loss import cons_Loss
import torch
from deepclustering2 import ModelMode
from deepclustering2.loss import SimplexCrossEntropyLoss
from deepclustering2.meters2 import UniversalDice, AverageValueMeter
from deepclustering2.models import ZeroGradientBackwardStep
from deepclustering2.schedulers import Weight_RampScheduler
from deepclustering2.utils import tqdm_, class2one_hot, Path, set_environment, write_yaml, flatten_dict, path2Path
from deepclustering2.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter


def average_list(input_list):
    return sum(input_list) / len(input_list)


class Trainer:
    PROJECT_PATH = str(Path(__file__).parents[1])
    RUN_PATH = str(Path(PROJECT_PATH, "runs"))

    def __init__(self,
                 model,
                 lab_loader: Union[DataLoader, _BaseDataLoaderIter],
                 unlab_loader: Union[DataLoader, _BaseDataLoaderIter],
                 val_loader: DataLoader,
                 weight_scheduler: Weight_RampScheduler = None,
                 weight_scheduler1: Weight_RampScheduler = None,
                 max_epoch: int = 100,
                 save_dir: str = "base",
                 checkpoint_path: str = None,
                 device="cpu",
                 config: dict = None,
                 num_batches=100,
                 *args,
                 **kwargs,
                 ):
        self._meter_interface = MeterInterface()
        self._model = model
        self._model.to(device=device)
        self._lab_loader = lab_loader
        self._unlab_loader = unlab_loader
        self._val_loader = val_loader
        self._weight_scheduler = weight_scheduler
        self._weight_scheduler1 = weight_scheduler1
        self._max_epoch = max_epoch
        self._start_epoch = 0
        self._best_score: float = -1
        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_path = checkpoint_path
        self._device = torch.device(device)
        self._num_batches = num_batches
        self.writer = SummaryWriter(str(self._save_dir))
        if config:
            self._config = config.copy()
            self._config.pop("Config", None)
            write_yaml(self._config, save_dir=self._save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))
        self._ce_criterion = SimplexCrossEntropyLoss()
        self.register_meters()

    def register_meters(self) -> None:
        c = self._config['Arch'].get('num_classes')
        report_axises = []
        for axi in range(c):
            report_axises.append(axi)

        self._meter_interface.register_meter(
            f"tra_dice", UniversalDice(C=c, report_axises=report_axises), group_name="train"
        )
        self._meter_interface.register_meter(
            f"val_dice", UniversalDice(C=c, report_axises=report_axises), group_name="val"
        )

        self._meter_interface.register_meter(
            "sup_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_meter(
            "reg_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_meter(
            "cons_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_meter(
            "total_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_meter(
            "lr", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_meter(
            "reg_weight", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_meter(
            "cons_weight", AverageValueMeter(), group_name="train"
        )

    def train_loop(self,
                   lab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                   unlab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                   epoch: int = 0,
                   mode=ModelMode.TRAIN,
                   *args,
                   **kwargs, ):
        self._model.set_mode(mode)
        batch_indicator = tqdm_(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")
        for batch_id, lab_data, unlab_data in zip(batch_indicator, lab_loader, unlab_loader):
            loss, vat_loss, cons_loss = self.runs(lab_data=lab_data, unlab_data=unlab_data)
            with ZeroGradientBackwardStep(
                    loss + self._weight_scheduler.value * vat_loss + self._weight_scheduler1.value * cons_loss,
                    self._model
            ) as new_loss:
                new_loss.backward()
            self._meter_interface["sup_loss"].add(loss.item())
            self._meter_interface["total_loss"].add(new_loss.item())

            try:
                self._meter_interface["reg_loss"].add(vat_loss.item())
            except Exception as e1:
                self._meter_interface["reg_loss"].add(vat_loss)

            try:
                self._meter_interface["cons_loss"].add(cons_loss.item())
            except Exception as e2:
                self._meter_interface["cons_loss"].add(cons_loss)

            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("train")
                batch_indicator.set_postfix(flatten_dict(report_statue))
        report_statue = self._meter_interface.tracking_status("train")
        batch_indicator.set_postfix(flatten_dict(report_statue))
        self.writer.add_scalar_with_tag(
            "train", flatten_dict(report_statue), global_step=epoch
        )

    def val_loop(self,
                 val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                 epoch: int = 0,
                 mode=ModelMode.EVAL,
                 *args,
                 **kwargs,
                 ):
        self._model.set_mode(mode)
        val_indicator = tqdm_(range(self._num_batches))
        val_indicator.set_description(f"Validation Epoch {epoch:03d}")
        for batch_id, data in enumerate(val_loader):
            image, target, filename = (
                data[0][0].to(self._device),
                data[0][1].to(self._device),
                data[1],
            )
            targetlv = torch.where(target < 3, torch.tensor([0]).to(self._device), torch.tensor([1]).to(self._device))
            preds = self._model(image).softmax(1)
            self._meter_interface["val_dice"].add(
                preds.max(1)[1],
                targetlv.squeeze(1),
                group_name=["_".join(x.split("_")[:-2]) for x in filename],
            )
            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("val")
                val_indicator.set_postfix(flatten_dict(report_statue))
        report_statue = self._meter_interface.tracking_status("val")
        val_indicator.set_postfix(flatten_dict(report_statue))
        self.writer.add_scalar_with_tag(
            "val", flatten_dict(report_statue), global_step=epoch
        )
        return average_list(self._meter_interface[f"val_dice"].summary().values())

    def StartTraining(self):
        for epoch in range(self._start_epoch, self._max_epoch):
            if self._model.get_lr() is not None:
                self._meter_interface["lr"].add(self._model.get_lr()[0])
            self._meter_interface["reg_weight"].add(self._weight_scheduler.value)
            self._meter_interface["cons_weight"].add(self._weight_scheduler1.value)
            self.train_loop(
                lab_loader=self._lab_loader,
                unlab_loader=self._unlab_loader,
                epoch=epoch
            )
            with torch.no_grad():
                current_score = self.val_loop(val_loader=self._val_loader, epoch=epoch)
            self.schedulerStep()
            self.save_checkpoint(self.state_dict(), epoch, current_score)

    def runs(self, lab_data, unlab_data):
        image, target, filename = (
            lab_data[0][0].to(self._device),
            lab_data[0][1].to(self._device),
            lab_data[1],
        )
        targetlv = torch.where(target < 3, torch.tensor([0]).to(self._device), torch.tensor([1]).to(self._device))
        onehot_target = class2one_hot(targetlv.squeeze(1), self._model._torchnet.num_classes)
        lab_preds = self._model(image).softmax(1)

        loss, reg_loss, cons_loss = 0.0, 0.0, 0.0

        loss = self._ce_criterion(lab_preds, onehot_target)
        if self._config['Train_vat']:
            vat_loss = VATLoss()
            uimage = unlab_data[0][0].to(self._device)
            reg_loss, unlab_preds_hat = vat_loss(self._model, uimage)
            if self._config['Constraints']['confident']:
                nondiff = cons_Loss()
                cons_loss = nondiff(unlab_preds_hat)  # constraints loss non-differentiable

        self._meter_interface["tra_dice"].add(
            lab_preds.max(1)[1],
            targetlv.squeeze(1),
            group_name=["_".join(x.split("_")[:-2]) for x in filename],
        )

        return loss, reg_loss, cons_loss

    def schedulerStep(self):
        self._model.schedulerStep()
        self._weight_scheduler.step()
        self._weight_scheduler1.step()

    def state_dict(self) -> Dict[str, Any]:
        """
        return trainer's state dict. The dict is built by considering all the submodules having `state_dict` method.
        """
        state_dictionary = {}
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict"):
                state_dictionary[module_name] = module.state_dict()
        return state_dictionary

    def save_checkpoint(
        self, state_dict, current_epoch, cur_score, save_dir=None, save_name=None
    ):
        save_best: bool = True if float(cur_score) > float(self._best_score) else False
        if save_best:
            self._best_score = float(cur_score)
        state_dict["epoch"] = current_epoch
        state_dict["best_score"] = float(self._best_score)
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
            if save_best:
                torch.save(state_dict, str(save_dir / "best.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))

    def _load_state_dict(self, state_dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        for module_name, module in self.__dict__.items():
            if hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError as e:
                    print(f"Loading checkpoint error for {module_name}, {e}.")
                except RuntimeError as e:
                    print(f"Interface changed error for {module_name}, {e}")

    def load_checkpoint(self, state_dict) -> None:
        """
        load checkpoint to models, meters, best score and _start_epoch
        Can be extended by add more state_dict
        :param state_dict:
        :return:
        """
        self._load_state_dict(state_dict)
        self._best_score = state_dict["best_score"]
        self._start_epoch = state_dict["epoch"] + 1

    def load_checkpoint_from_path(self, checkpoint_path):
        checkpoint_path = path2Path(checkpoint_path)
        assert checkpoint_path.exists(), checkpoint_path
        if checkpoint_path.is_dir():
            state_dict = torch.load(
                str(Path(checkpoint_path) / self.checkpoint_identifier),
                map_location=torch.device("cpu"),
            )
        else:
            assert checkpoint_path.suffix == ".pth", checkpoint_path
            state_dict = torch.load(
                str(checkpoint_path), map_location=torch.device("cpu"),
            )
        self.load_checkpoint(state_dict)