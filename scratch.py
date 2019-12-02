from toblerone import projection
import toblerone as tob 
from pdb import set_trace

if __name__ == "__main__":
    
    ins = 'testdata/103818.L.white.32k_fs_LR.surf.gii'
    outs = 'testdata/103818.L.pial.32k_fs_LR.surf.gii'
    spc = 'testdata/ref_2.6.nii.gz'
    img = 'testdata/ones.nii.gz'
    factor = 5
    cores = 16 

    # out = projection.vol2surf(img, ins, outs, spc, 5)

    ins = tob.Surface(ins)
    outs = tob.Surface(outs)
    spc = tob.ImageSpace(spc)

    ins.applyTransform(spc.world2vox)
    outs.applyTransform(spc.world2vox)
    # x = projection._form_vtxtri_mat(ins)
    x = projection.vol2prism_weights(ins, outs, spc, factor, 16)

    set_trace()
    print('dibe')