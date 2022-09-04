import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys

from utils import hist_clipping, create_unimodal_mask

logging.basicConfig(level=logging.INFO)


voxel_size = gp.Coordinate((1,) * 2)

shape = gp.Coordinate((256,) * 2)

num_samples = 5
batch_size = 10


class Normalize(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def setup(self):

        spec = self.spec[self.array].copy()

        spec.dtype = np.float32

        self.updates(self.array, spec)

    def process(self, batch, request):

        data = batch[self.array].data

        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = data.astype(np.float32)

        batch[self.array].data = data
        batch[self.array].spec.dtype = np.float32


class HistClip(gp.BatchFilter):
    def __init__(self, array, min_percentile, max_percentile):
        self.array = array
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def setup(self):

        spec = self.spec[self.array].copy()

        spec.dtype = np.float32

        self.updates(self.array, spec)

    def process(self, batch, request):

        data = batch[self.array].data

        data = hist_clipping(
                data,
                self.min_percentile,
                self.max_percentile).astype(np.float32)

        batch[self.array].data = data
        batch[self.array].spec.dtype = np.float32


class CreateMask(gp.BatchFilter):
    def __init__(self, in_array, out_array):
        self.in_array = in_array
        self.out_array = out_array

    def setup(self):

        self.provides(self.out_array, self.spec[self.in_array].copy())

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.in_array] = request[self.out_array].copy()

        return deps

    def process(self, batch, request):

        data = batch[self.in_array].data

        mask = create_unimodal_mask(data).astype(np.uint8)

        spec = batch[self.in_array].spec.copy()
        spec.roi = request[self.out_array].roi.copy()
        spec.dtype = np.uint8

        batch = gp.Batch()

        batch[self.out_array] = gp.Array(mask, spec)

        return batch


def pipeline(iterations):

    actin = gp.ArrayKey("ACTIN")
    dna = gp.ArrayKey("DNA")
    phase = gp.ArrayKey("PHASE")
    ret = gp.ArrayKey("RET")
    mask = gp.ArrayKey("MASK")

    request = gp.BatchRequest()
    request.add(actin, shape)
    request.add(dna, shape)
    request.add(phase, shape)
    request.add(ret, shape)
    request.add(mask, shape)

    sources = tuple(
        gp.ZarrSource(
            "data/simple_test.zarr",
            datasets={
                actin: f"actin/{i}",
                dna: f"dna/{i}",
                phase: f"phase/{i}",
                ret: f"ret/{i}",
            },
            array_specs={
                actin: gp.ArraySpec(interpolatable=True),
                dna: gp.ArraySpec(interpolatable=True),
                phase: gp.ArraySpec(interpolatable=True),
                ret: gp.ArraySpec(interpolatable=True),
            },
        )
        # HistClip(actin, 1, 99) +
        # HistClip(dna, 1, 99) +
        # HistClip(phase, 0.8, 99.5) +
        # HistClip(ret, 0.8, 99.5) +
        + Normalize(actin)
        + Normalize(dna)
        + Normalize(phase)
        + Normalize(ret)
        + gp.RandomLocation()
        for i in range(num_samples)
    )

    pipeline = sources

    pipeline += gp.RandomProvider()
    pipeline += CreateMask(dna, mask)

    # this is just a very hacky way to visualize the non augmented vs augmented
    # data on a single batch...

    # not augmented
    snap = gp.Snapshot(
        output_filename="batch_{iteration}.zarr",
        dataset_names={
            actin: "no_aug_actin",
            dna: "no_aug_dna",
            phase: "no_aug_phase",
            ret: "no_aug_ret",
            mask: "no_aug_mask",
        },
        every=1,
    )

    # snapshot is write by default but we want to just append to it so we don't
    # wipe our non augmented datasets
    # this means if we try to rerun this script again after it will fail since
    # the zarr will contain data already, so you would just remove the batch.
    # but this should definitely just be taken as an example of how you could go
    # about this, you should definitely do this in a better way
    snap.mode = 'a'

    pipeline += snap

    pipeline += gp.ElasticAugment(
        control_point_spacing=(64, 64),
        jitter_sigma=(4, 4),
        rotation_interval=(0, math.pi / 2),
    )

    pipeline += gp.SimpleAugment()

    # pipeline += CreateMask(dna, mask)

    # pipeline += gp.Unsqueeze([actin, dna, phase, ret, mask])
    # pipeline += gp.Stack(batch_size)

    # now add our augmented samples for the same batch
    snap = gp.Snapshot(
        output_filename="batch_{iteration}.zarr",
        dataset_names={
            actin: "aug_actin",
            dna: "aug_dna",
            phase: "aug_phase",
            ret: "aug_ret",
            mask: "aug_mask",
        },
        every=1,
    )

    snap.mode = 'a'

    pipeline += snap

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)


if __name__ == "__main__":

    pipeline(1)
