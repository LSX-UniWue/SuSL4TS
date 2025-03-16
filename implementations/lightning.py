from typing import Dict, Any

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection

from susl_base.networks.gmm_dgm import GaussianMixtureDeepGenerativeModel
from susl_base.networks.lightning import LightningGMMModel
from susl_base.networks.losses import GaussianMixtureDeepGenerativeLoss


class LightningGMMModelWeightDecay(LightningGMMModel):
    def __init__(
        self,
        model: GaussianMixtureDeepGenerativeModel,
        loss_fn: GaussianMixtureDeepGenerativeLoss,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        lr: float = 1e-3,
        loss_fn_step_step_size: float = 0.02,
        loss_fn_step_max_value: float = 1.0,
        weight_decay: float = 1e-5,
        cosine_t_max: int = 100,
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            loss_fn_step_max_value=loss_fn_step_max_value,
            loss_fn_step_step_size=loss_fn_step_step_size,
        )
        self.__optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.__cosine_t_max = cosine_t_max

    def configure_optimizers(self) -> Dict[str, Any]:
        return {
            "optimizer": self.__optimizer,
            "lr_scheduler": CosineAnnealingLR(optimizer=self.__optimizer, T_max=self.__cosine_t_max),
        }
