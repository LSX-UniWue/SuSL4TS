from typing import Literal, Tuple

from numpy import genfromtxt, expand_dims, concatenate
from torch import Tensor, from_numpy
from torch.utils.data import Dataset


class HARDataset(Dataset):
    features = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]
    classes = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    resource = "datasets_processed/UCI_HAR_Dataset"
    mean = [
        -5.5118e-04,
        -3.4515e-04,
        -3.9009e-04,
        -1.8340e-03,
        -5.7291e-04,
        2.6344e-04,
        8.0239e-01,
        3.0805e-02,
        8.7386e-02,
    ]
    std = [0.1951, 0.1224, 0.1061, 0.4053, 0.3773, 0.2549, 0.4166, 0.3919, 0.3590]

    def __init__(self, stage: Literal["train", "test"] = "train") -> None:
        super().__init__()
        x_data = []
        for feature in self.features:
            tmp = genfromtxt(f"{self.resource}/{stage}/InertialSignals/{feature}_{stage}.txt")
            x_data.append(expand_dims(tmp, axis=1))
        self.__x_data = concatenate(x_data, axis=1).astype("float32")
        # Ground truth is 1-indexed
        self.__y_data = genfromtxt(f"{self.resource}/{stage}/y_{stage}.txt", dtype="uint8") - 1

    def __len__(self) -> int:
        return self.__y_data.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        x = from_numpy(self.__x_data[idx])
        y = self.__y_data[idx]
        return x, y
