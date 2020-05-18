# Quickstart

Toblerone is a set of tools for surface-based analysis. In particular, it can perform partial volume estimation from surface segmentations (for example, those provided by FreeSurfer). It can be used either within python scripts or at the command line. 

## Python scripting

The python interface provides more control over the individual functions and also allows you to assemble your own PV estimation framework if the standard ones are not suitable. The key functions are: 

- `toblerone.pvestimation.structure` Estimate PVs arising from a structure defined by a single surface. Structures defined by multiple surfaces can be processed by estimating on each individual surface and subtracting the results (as is done in `estimate_cortex`). This returns an array of size equal to the reference image space, where the value in each position corresponds to the proportion of that voxel that is *interior* to the structure. 

- `toblerone.pvestimation.cortex` Estimate PVs within the cortical ribbon (either one or both hemispheres). If both hemispheres are requested (default) then the results from each will be combined into a single image. Returns an array of size equal to the reference, extended to three volumes in the 4th dimension (eg X Y Z 3), where the three values in each position give the GM, WM and non-brain (CSF + background + everything else) PVs respectively.  

### Example usage (for the cortex)

```python 
import toblerone 

# FreeSurfer native surfaces
LWS = 'path/to/lh.white'
LPS = 'path/to/lh.pial'
RPS = 'path/to/rh.pial'
RWS = 'path/to/rh.white'
ref = 'path/to/functional.nii.gz'

# NB if using a FSL FLIRT matrix also set 'flirt=True' and 
# provide a path to structural image below
struct2ref = 'registration_structural_to_reference.mat'

# This returns a 4D array, sized equal to the reference image, 
# with the PVs arranged GM, WM and non-brain in the last dimension 
pvs = toblerone.pvestimation.cortex(LWS=LWS, LPS=LPS, RPS=RPS, RWS=RWS, struct2ref=struct2ref, ref=ref)
```

## Command line 

The following functions and utilities are available: 

`toblerone -estimate_all`: estimate partial volumes across the brain, both cortex and subcortex 

`toblerone -estimate_cortex`: just the cortex, one or both hemispheres. 

`toblerone -estimate_structure`: just a structure defined by a single surface (eg thalamus)

`toblerone -fs_fs_anat`: pre-processing script to run FSL's `fsl_anat` and FreeSurfer's `recon-all` at default settings to obtain cortical and subcortical surfaces

`convert-surface`: surface conversion utility. **NB FreeSurfer surfaces will have the C_ras coordinate shift applied to them automatically**