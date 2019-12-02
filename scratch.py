from toblerone import projection
import toblerone as tob 
from pdb import set_trace
import numpy as np 
import nibabel as nib 

if __name__ == "__main__":
    
    ins = 'testdata/103818.L.white.32k_fs_LR.surf.gii'
    outs = 'testdata/103818.L.pial.32k_fs_LR.surf.gii'
    spc = 'testdata/ref_2.6.nii.gz'

    ins = tob.Surface(ins)
    outs = tob.Surface(outs)
    spc = tob.ImageSpace(spc)
    
    # If you need to apply a registration then do it here, but the surfaces
    # need to remain in world-mm coordinates
    # surface.applyTransform(affine)

    # Produce a volume of ones, then map it to the surface and back again 
    voldata = np.ones(spc.size.prod(), np.float32)
    smapped = projection.vol2surf(voldata, ins, outs, spc)
    vmapped = projection.surf2vol(smapped, ins, outs, spc)

    # Mask to voxels that had a value mapped back to them (voxels 
    # with no surface will get a zero value)
    # Of those, get the largest and smallest values (should be close to 1)
    mask = (vmapped > 0)
    print(vmapped[mask].min(), vmapped[mask].max())

    # Save as gifti
    da = nib.gifti.GiftiDataArray(smapped)
    gii = nib.gifti.GiftiImage(darrays=[da])
    nib.save(gii, 'testdata/smapped.func.gii')
    spc.save_image(vmapped, 'testdata/vmapped.nii.gz')
