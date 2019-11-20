from toblerone import projection

if __name__ == "__main__":
    
    ins = 'testdata/103818.L.white.32k_fs_LR.surf.gii'
    outs = 'testdata/103818.L.pial.32k_fs_LR.surf.gii'
    spc = 'testdata/ref_2.6.nii.gz'
    img = 'testdata/ones.nii.gz'

    out = projection.vol2surf(img, ins, outs, spc, 5)