from pathlib import Path
from typing import Callable, Optional, Union, Type
import importlib
import lightning as L
import torch
import torch.utils
from data import CombinedSliceDataset, VolumeSampler, FastmriSliceDataset
from data import InferVolumeBatchSampler, InferVolumeDistributedSampler


#########################################################################################################
# Common functions
#########################################################################################################

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        torch.utils.data.Dataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))

def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False

def resolve_class(class_path: str):
    """Dynamically resolve a class from its string path."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

#########################################################################################################
# DataModule
#########################################################################################################

class DataModule(L.LightningDataModule):
    """
    Base data module class for MRI datasets.
    
    This class handles common configurations for training on MRI data. Specific dataset
    implementations should inherit from this class and override/implement necessary methods.
    """


    def __init__(
        self,
        slice_dataset: str,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        combine_train_val: bool = False,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        num_adj_slices: int = 5,
        data_balancer: Optional[Callable] = None,
        use_pre_whiten: bool = False,
    ):
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError("Can set val_sample_rate or val_volume_sample_rate, but not both.")

        self.slice_dataset = resolve_class(slice_dataset)
        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.combine_train_val = combine_train_val
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.num_adj_slices = num_adj_slices
        self.data_balancer = data_balancer
        self.use_pre_whiten = use_pre_whiten
    def _create_data_loader(
        self,
        slice_dataset: Type,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        raw_sample_filter: Optional[Callable]
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter

        # if desired, combine train and val together for the train split
        dataset: Union[slice_dataset, CombinedSliceDataset] #TODO: use base slice class
        prefix = f"{self.challenge}_" if self.slice_dataset is FastmriSliceDataset else ""
        if is_train and self.combine_train_val:

            data_paths = [
                self.data_path / f"{prefix}train",
                self.data_path / f"{prefix}val",
            ]

            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                slice_dataset= slice_dataset,
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                data_balancer=self.data_balancer,
                num_adj_slices = self.num_adj_slices,
                use_pre_whiten=self.use_pre_whiten,
            )
        else:
            data_path = self.data_path / f"{prefix}{data_partition}"

            dataset = slice_dataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                data_balancer=self.data_balancer,
                num_adj_slices = self.num_adj_slices,
                use_pre_whiten=self.use_pre_whiten,
            )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers= self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            prefix = f"{self.challenge}_" if self.slice_dataset is FastmriSliceDataset else ""

            data_paths = [
                self.data_path / f"{prefix}train",
                self.data_path / f"{prefix}val",
            ]

            data_transforms = [
                self.train_transform,
                self.val_transform,
            ]
            
            raw_sample_filters = [
                self.train_filter,
                self.val_filter,
            ]
            
            data_balancers = [
                self.data_balancer,
                None,
            ]
            
            for i, (data_path, data_transform, raw_sample_filter, data_balancer) in enumerate(
                zip(data_paths, data_transforms, raw_sample_filters, data_balancers)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                _ = self.slice_dataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                    raw_sample_filter=raw_sample_filter,
                    num_adj_slices = self.num_adj_slices,
                    data_balancer=data_balancer,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.val_transform, data_partition="val")
   
    def predict_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.val_transform, data_partition="val")


#########################################################################################################
# Inference DataModule
#########################################################################################################

class InferenceDataModule(L.LightningDataModule):
    """
    Base data module class for MRI datasets.
    
    This class handles common configurations for training on MRI data. Specific dataset
    implementations should inherit from this class and override/implement necessary methods.
    """

    def __init__(
        self,
        slice_dataset: str,
        data_path: Path,
        test_transform: Callable,
        challenge: str="multicoil",
        test_filter: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        num_adj_slices: int = 5,
    ):
        super().__init__()

        self.slice_dataset = resolve_class(slice_dataset)
        self.data_path = data_path
        self.challenge = challenge
        self.test_transform = test_transform
        self.test_filter = test_filter

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.num_adj_slices = num_adj_slices

    def _create_data_loader(
        self,
        slice_dataset: Type,
        data_transform: Callable,
        data_path: Path,
    ) -> torch.utils.data.DataLoader:
        
        raw_sample_filter = self.test_filter

        dataset = slice_dataset(
            root=data_path,
            transform=data_transform,
            challenge=self.challenge,
            raw_sample_filter=raw_sample_filter,
            num_adj_slices = self.num_adj_slices
        )

        # ensure that entire volumes go to the same GPU in the ddp setting
        
        if self.distributed_sampler:
            sampler = InferVolumeDistributedSampler(dataset, shuffle=False)
            batch_sampler = InferVolumeBatchSampler(
                sampler,
                batch_size=self.batch_size,
                drop_last=False,
                index_to_volume_idx=dataset.index_to_volume_idx,
            )
            # if use sampler, do not change default batch size and shuffle
            batch_size = 1 
            shuffle = None 
        else:
            batch_size = self.batch_size
            batch_sampler = None
            shuffle = False
    
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers= self.num_workers,
            worker_init_fn=worker_init_fn,
            batch_sampler=batch_sampler,
            shuffle=shuffle,
        )

        return dataloader


    def predict_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.test_transform, data_path=self.data_path)





