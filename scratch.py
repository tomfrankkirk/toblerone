from toblerone import projection
import toblerone as tob 
from pdb import set_trace
import numpy as np 

if __name__ == "__main__":
    
    ins = 'testdata/103818.L.white.32k_fs_LR.surf.gii'
    outs = 'testdata/103818.L.pial.32k_fs_LR.surf.gii'
    spc = 'testdata/ref_2.6.nii.gz'
    img = 'testdata/ones.nii.gz'
    factor = 10
    cores = 8

    ins = tob.Surface(ins)
    outs = tob.Surface(outs)
    spc = tob.ImageSpace(spc)
    v2s_weights = projection.vol2surf_weights(ins, outs, spc, factor, cores)
    s2v_weights = projection.surf2vol_weights(ins, outs, spc, factor, cores)

    voldata = np.ones(spc.size.prod(), np.float32)
    for _ in range(10):
        smapped = v2s_weights.dot(voldata)
        vmapped = s2v_weights.dot(smapped)

    mask = (vmapped > 0)

    surfdata = np.ones(ins.points.shape[0], np.float32)
    volmapped = s2v_weights.dot(surfdata)

    set_trace()
    print('dibe')