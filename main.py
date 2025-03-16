from functools import partial

from lightning import Trainer
from torch import Generator, Tensor, tensor
from torch.nn import Sequential, ReLU, Identity, Flatten, Upsample, Linear, Conv1d
from torch.utils.data import random_split
from torchmetrics import MetricCollection

from datasets.mitbh_dataset import MITBIHDataset
from implementations.lightning import LightningGMMModelWeightDecay
from implementations.susl_dataset import TransformableLabeledDatasetFacade, TransformableDatasetFacade
from implementations.variational_layer import MSEVariationalLayer
from susl_base.data.data_module import SemiUnsupervisedDataModule
from susl_base.data.utils import create_susl_dataset
from susl_base.metrics.cluster_and_label import ClusterAccuracy
from susl_base.networks.gmm_dgm import EntropyRegularizedGaussianMixtureDeepGenerativeModel
from susl_base.networks.latent_layer import LatentLayer
from susl_base.networks.losses import EntropyGaussianMixtureDeepGenerativeLoss
from susl_base.networks.misc import Reshape
from susl_base.networks.variational_layer import GaussianVariationalLayer


# Copied from susl_base.main
def get_prior(n_l: int, n_aug: int) -> Tensor:
    from torch import tensor

    # Unsupervised
    if n_l <= 0:
        return tensor(n_aug * [1 / n_aug])
    # (Semi-)Supervised
    if n_aug <= 0:
        return tensor(n_l * [1 / n_l])
    # SuSL
    return 0.5 * tensor(n_l * [1 / n_l] + n_aug * [1 / n_aug])


# z-norm
def input_transform(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (x - mean) / std


# Return raw target
def target_transform(x: Tensor) -> Tensor:
    return x


def run_cnn() -> None:
    train_dataset, validation_dataset = random_split(
        MITBIHDataset(stage="train"),
        lengths=[0.8, 0.2],
        generator=Generator().manual_seed(42),
    )
    test_dataset = MITBIHDataset(stage="test")

    partial_input_transform = partial(input_transform, mean=tensor(test_dataset.mean), std=tensor(test_dataset.std))
    labeled_dataset_facade_init = partial(
        TransformableLabeledDatasetFacade, input_transform=partial_input_transform, target_transform=target_transform
    )
    dataset_facade_init = partial(
        TransformableDatasetFacade, input_transform=partial_input_transform, target_transform=target_transform
    )
    train_dataset_labeled, train_dataset_unlabeled, class_mapper = create_susl_dataset(
        dataset=train_dataset,
        num_labels=0.2,
        classes_to_hide=[0],
        labeled_dataset_facade_init=labeled_dataset_facade_init,
        dataset_facade_init=dataset_facade_init,
    )

    # Create model
    n_l, n_aug, n_classes = 4, 10, 5
    n_x, n_y, n_z = 186, n_l + n_aug, 50
    datamodule = SemiUnsupervisedDataModule(
        train_dataset_labeled=train_dataset_labeled,
        train_dataset_unlabeled=train_dataset_unlabeled,
        validation_dataset=TransformableLabeledDatasetFacade(
            validation_dataset,
            indices=list(range(len(validation_dataset))),
            class_mapper=class_mapper,
            input_transform=partial_input_transform,
            target_transform=target_transform,
        ),
        test_dataset=TransformableLabeledDatasetFacade(
            test_dataset,
            indices=list(range(len(test_dataset))),
            class_mapper=class_mapper,
            input_transform=partial_input_transform,
            target_transform=target_transform,
        ),
        batch_size=512,
    )

    q_y_x_module = Sequential(
        Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Flatten(),
        Linear(in_features=32 * 12 * 1, out_features=n_y),
    )
    p_x_z_module = MSEVariationalLayer(
        feature_extractor=Sequential(
            Linear(in_features=n_z, out_features=32 * 23 * 1),
            Reshape((-1, 32, 23)),
            Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
            Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
            Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
        ),
        module_init=Conv1d,
        out_channels=1,
        in_channels=1,
        kernel_size=1,
        padding=1,
    )
    p_z_y_module = GaussianVariationalLayer(feature_extractor=Identity(), in_features=n_y, out_features=n_z)
    q_z_xy_module = GaussianVariationalLayer(
        feature_extractor=LatentLayer(
            pre_module=Sequential(
                Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Flatten(),
            ),
            post_module=Sequential(Linear(in_features=32 * 12 * 1 + n_y, out_features=128), ReLU()),
        ),
        out_features=n_z,
        in_features=128,
    )
    model = EntropyRegularizedGaussianMixtureDeepGenerativeModel(
        n_y=n_y,
        n_z=n_z,
        n_x=n_x,
        q_y_x_module=q_y_x_module,
        p_x_z_module=p_x_z_module,
        p_z_y_module=p_z_y_module,
        q_z_xy_module=q_z_xy_module,
        log_priors=get_prior(n_l=n_l, n_aug=n_aug).log(),
    )
    print(model)
    # Create trainer and run
    lt_model = LightningGMMModelWeightDecay(
        model=model,
        loss_fn=EntropyGaussianMixtureDeepGenerativeLoss(),
        val_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_classes, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_classes, average="macro"),
            },
            prefix="val_",
        ),
        test_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_classes, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_classes, average="macro"),
            },
            prefix="test_",
        ),
        weight_decay=1e-5,
        cosine_t_max=100,
    )
    trainer = Trainer(max_epochs=100, check_val_every_n_epoch=2)
    trainer.fit(model=lt_model, datamodule=datamodule)
    trainer.test(model=lt_model, datamodule=datamodule)


if __name__ == "__main__":
    run_cnn()
