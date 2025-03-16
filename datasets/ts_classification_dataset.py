from typing import Literal, Tuple

from numpy import loadtxt
from torch import Tensor, from_numpy
from torch.utils.data import Dataset


class TSClassificationDataset(Dataset):
    classes = None
    resource = "datasets_processed/Univariate_arff"
    name = None
    mean = 0.0
    std = 1.0

    def __init__(self, stage: Literal["train", "test"] = "train") -> None:
        super().__init__()
        tmp = loadtxt(f"{self.resource}/{self.name}/{self.name}_{stage.upper()}.txt")
        self.__x_data = tmp[:, 1:].astype("float32")
        # Ground truth is 1-indexed
        self.__y_data = tmp[:, 0].astype("int64") - 1

    def __len__(self) -> int:
        return self.__y_data.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        x = from_numpy(self.__x_data[idx]).unsqueeze(0)
        y = self.__y_data[idx]
        return x, y


class ElectricDevicesDataset(TSClassificationDataset):
    name = "ElectricDevices"
    classes = ["1", "2", "3", "4", "5", "6", "7", "8"]  # Could not find proper class names in the publication
