from torch import Tensor, tensor
from torch.nn import Module, Conv1d

from susl_base.networks.variational_layer import VariationalLayer


class MSEVariationalLayer(VariationalLayer):
    def __init__(self, feature_extractor: Module, module_init=Conv1d, **kwargs) -> None:
        VariationalLayer.__init__(self, feature_extractor)
        self.__linear = module_init(**kwargs)
        self.register_buffer("values", tensor=tensor(0.0), persistent=False)

    def forward(self, x: Tensor, y: Tensor = None) -> None:
        latent = VariationalLayer.forward(self, x, y)
        self.values = self.__linear(latent)

    # For consistency with susl_base -> log_prob returns -mse
    def log_prob(self, value: Tensor) -> Tensor:
        return -(self.values - value).pow(2)
