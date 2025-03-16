from typing import Literal, Tuple

from numpy import genfromtxt
from torch import Tensor, from_numpy
from torch.utils.data import Dataset


class MITBIHDataset(Dataset):
    classes = ["N", "S", "V", "F", "Q"]
    resource = "datasets_processed/kaggle_ecg_classification"
    mean = 0.175
    std = 0.227

    def __init__(self, stage: Literal["train", "test"] = "train") -> None:
        super().__init__()
        tmp = genfromtxt(f"{self.resource}/mitbih_{stage}.csv", delimiter=",")
        self.__x_data, self.__y_data = tmp[:, :-2].astype("float32"), tmp[:, -1].astype("uint8")

    def __len__(self) -> int:
        return self.__y_data.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        x = from_numpy(self.__x_data[idx]).unsqueeze(0)
        y = self.__y_data[idx]
        return x, y
