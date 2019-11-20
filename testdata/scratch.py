if __name__ == "__main__":
    
    ins = '103818.L.white.32k_fs_LR.surf.gii'
    outs = '103818.L.pial.32k_fs_LR.surf.gii'
    spc = 'ref_2.6.nii.gz'
    img = 'ones.nii.gz'

    out = vol2surf(img, ins, outs, spc, 5)