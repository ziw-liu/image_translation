import neuroglancer
import numpy as np
import os
import sys
import zarr

neuroglancer.set_server_bind_address('localhost')


# very lazy way to view the non augmented and augmented data, this script should
# be refactored of course

f = zarr.open(sys.argv[1])

all_datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]
no_aug_datasets = sorted([i for i in os.listdir(sys.argv[1]) if 'no_aug' in i])
aug_datasets = sorted([i for i in all_datasets if i not in no_aug_datasets])

res = f[all_datasets[0]].attrs['resolution']

viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['b','c^','y','x'],
        units='nm',
        scales=res+res)

with viewer.txn() as s:

    for aug, no_aug in zip(aug_datasets, no_aug_datasets):

        offset = f[aug].attrs['offset']

        offset = [0,]*2 + [int(i/j) for i,j in zip(offset, res)]

        aug_data = f[aug][:]
        no_aug_data = f[no_aug][:]

        aug_data = np.expand_dims(np.expand_dims(aug_data, axis=0), axis=0)
        no_aug_data = np.expand_dims(np.expand_dims(no_aug_data, axis=0), axis=0)

        if 'mask' in aug:
            aug_data = aug_data.astype(np.float32)
        if 'mask' in no_aug:
            no_aug_data = no_aug_data.astype(np.float32)


        shader="""
void main() {
emitRGB(
    vec3(
        toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue(2)))
    );
}"""

        s.layers[aug] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=aug_data,
                voxel_offset=offset,
                dimensions=dims),
            shader=shader)

        s.layers[no_aug] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=no_aug_data,
                voxel_offset=offset,
                dimensions=dims),
            shader=shader)

    s.layout = neuroglancer.column_layout(
            [
            neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(
                        layers=[ds_name],
                        layout='yz') for ds_name in no_aug_datasets
                ]
            ),
            neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(
                        layers=[ds_name],
                        layout='yz') for ds_name in aug_datasets
                ]
            )])


print(viewer)
