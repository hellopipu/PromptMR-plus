"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import random 
from torch.utils.data import DistributedSampler



class VolumeSampler(Sampler):
    """
    Sampler for volumetric MRI data.

    Based on pytorch DistributedSampler, the difference is that all instances
    from the same MRI volume need to go to the same node for distributed
    training. Dataset example is a list of tuples (fname, instance), where
    fname is essentially the volume name (actually a filename).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset, #Union[CombinedSliceDataset, SliceDataset],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset: An MRI dataset (e.g., SliceData).
            num_replicas: Number of processes participating in distributed
                training. By default, :attr:`rank` is retrieved from the
                current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`. By
                default, :attr:`rank` is retrieved from the current distributed
                group.
            shuffle: If ``True`` (default), sampler will shuffle the indices.
            seed: random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across
                all processes in the distributed group.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # get all file names and split them based on number of processes
        self.all_volume_names = sorted(
            set(str(raw_sample[0]) for raw_sample in self.dataset.raw_samples)
        )
        self.all_volumes_split: List[List[str]] = []
        for rank_num in range(self.num_replicas):
            self.all_volumes_split.append(
                [
                    self.all_volume_names[i]
                    for i in range(
                        rank_num, len(self.all_volume_names), self.num_replicas
                    )
                ]
            )

        # get slice indices for each file name
        rank_indices: List[List[int]] = [[] for _ in range(self.num_replicas)]
        for i, raw_sample in enumerate(self.dataset.raw_samples):
            vname = str(raw_sample[0])
            for rank_num in range(self.num_replicas):
                if vname in self.all_volumes_split[rank_num]:
                    rank_indices[rank_num].append(i)
                    break

        # need to send equal number of samples to each process - take the max
        self.num_samples = max(len(indices) for indices in rank_indices)
        self.total_size = self.num_samples * self.num_replicas
        self.indices = rank_indices[self.rank]

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ordering = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in ordering]
        else:
            indices = self.indices

        # Check for empty indices to avoid division by zero
        if len(indices) == 0:
            print(f"Warning: Empty indices for rank {self.rank}. Returning empty iterator.")
            return iter([])

        # add extra samples to match num_samples
        repeat_times = self.num_samples // len(indices)
        indices = indices * repeat_times
        indices = indices + indices[: self.num_samples - len(indices)]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        # print('debug sampler: ', self.rank, self.num_samples)
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InferVolumeDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            if not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        # Build a list of volume indices
        self.volume_indices = dataset.volume_indices

        # Partition volumes among replicas
        self.volumes_per_replica = self._partition_volumes()

    def _partition_volumes(self):
        volumes_per_replica = [[] for _ in range(self.num_replicas)]
        for idx, volume in enumerate(self.volume_indices):
            replica_idx = idx % self.num_replicas
            volumes_per_replica[replica_idx].extend(volume)
        return volumes_per_replica[self.rank]

    def __iter__(self):
        indices = self.volumes_per_replica
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.volumes_per_replica)

class InferVolumeBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, index_to_volume_idx):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.index_to_volume_idx = index_to_volume_idx

    def __iter__(self):
        batch = []
        prev_volume_idx = None

        for idx in self.sampler:
            volume_idx = self.index_to_volume_idx[idx]
            if prev_volume_idx is not None and volume_idx != prev_volume_idx:
                if batch:
                    yield batch
                    batch = []
            batch.append(idx)
            prev_volume_idx = volume_idx
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        # Calculate the number of batches
        total_samples = len(self.sampler)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size