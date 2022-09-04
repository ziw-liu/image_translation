import neuroglancer
import numpy as np
import os
import sys
import zarr

# simple view script with neuroglancer. see ../view_batch.py for more thorough
# overview

neuroglancer.set_server_bind_address('localhost')

f = 'simple_test.zarr'

container = zarr.open(f)

datasets = [i for i in os.listdir(f) if '.' not in i]

viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['t','y','x'],
        units='nm',
        scales=[1]*3)

with viewer.txn() as s:

    for ds in datasets:
        sections = sorted([i for i in os.listdir(os.path.join(f, ds)) if '.' not in i])

        full_data = []

        for sec in sections:

            sec_data = container[f'{ds}/{sec}'][:]

            sec_data = (sec_data - np.min(sec_data)) / (np.max(sec_data) - np.min(sec_data))
            sec_data = sec_data.astype(np.float32)

            full_data.append(sec_data)

        full_data = np.array(full_data)

        s.layers[ds] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=full_data,
                voxel_offset=[0]*3,
                dimensions=dims))

        # break

print(viewer)
