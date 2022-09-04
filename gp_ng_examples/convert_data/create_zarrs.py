import glob
import numpy as np
import zarr

from skimage.io import imread
from natsort import natsorted

postion = 3

# get 5 sections at position 3
actin = natsorted(glob.glob(f'data/img_568_t000_p00{postion}*'))
dna = natsorted(glob.glob(f'data/img_405_t000_p00{postion}*'))
phase = natsorted(glob.glob(f'data/img_phase_t000_p00{postion}*'))
ret = natsorted(glob.glob(f'data/img_Retardance_t000_p00{postion}*'))

# create out zarr
out = zarr.open('simple_test.zarr', 'a')

for ds_name, data in [
        ('actin', actin),
        ('dna', dna),
        ('phase', phase),
        ('ret', ret)]:

    for idx, sec in enumerate(data):

        # load data to np array
        sec_data = imread(sec)

        # write to our zarr with arbitrary offset and resolution
        out[f'{ds_name}/{idx}'] = sec_data
        out[f'{ds_name}/{idx}'].attrs['offset'] = [0]*2
        out[f'{ds_name}/{idx}'].attrs['resolution'] = [1]*2


