import neuroglancer
import numpy as np
import os
import sys
import zarr

# example batch viewing script

# if you are serving over ssh change to remote host ip
neuroglancer.set_server_bind_address('localhost')

f = zarr.open(sys.argv[1])

# get all of our batch datasets - make sure we don't fetch a hidden .zgroup
datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]

# get resolution
res = f[datasets[0]].attrs['resolution']

# create viewer
viewer = neuroglancer.Viewer()

# tell neuroglancer about our data (this should match the output shapes from the
# pipeline. names without a carat will be rendered as spatial dimensions, names with a
# carat will be rendered as channel dimensions (so the below will allow us to
# see our data with a shader over the channel dimension, but spatially we will
# see batch, y, x and scroll through the batches. scales and offset shapes need
# to be compatible with data shapes
dims = neuroglancer.CoordinateSpace(
        names=['b','c^','y','x'],
        units='nm',
        scales=res+res)

# create session
with viewer.txn() as s:

    for ds in datasets:

        # get offset
        offset = f[ds].attrs['offset']

        offset = [0,]*2 + [int(i/j) for i,j in zip(offset, res)]

        # load to np array
        data = f[ds][:]

        # just for visualizing
        if ds == 'mask':
            data = data.astype(np.float32)

        # glsl shader
        shader="""
void main() {
emitRGB(
    vec3(
        toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue(2)))
    );
}"""

        # add as layers. LocalVolume gets us from numpy to neuroglancer land.
        # ImageLayer should be used since we don't have segmentations (if we
        # did we would use SegmentationLayer
        s.layers[ds] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=data,
                voxel_offset=offset,
                dimensions=dims),
            shader=shader)

    # change the layout to render each dataset as a column by default
    s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(
                    layers=[ds_name],
                    layout='yz') for ds_name in datasets
            ]
        )
# print the url (run in interactive to keep the session alive),
# eg python -i view_batch.py
print(viewer)
