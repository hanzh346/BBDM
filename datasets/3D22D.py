import os
import nibabel as nib
from PIL import Image
from pathlib import Path

# Path to the folder containing your 3D NIfTI files
nifti_dir = "/home/hang/GitHub/BBDM/datasets/low_dose_PET/test/A"

# Path where you want to save the slices
output_dir = "/home/hang/GitHub/BBDM/datasets/low_dose_PET_2D/test/A"

# Iterate over all NIfTI files in the folder
for nifti_file in Path(nifti_dir).glob("*.nii"):
    # Load the NIfTI file
    nii = nib.load(str(nifti_file))
    image = nii.get_fdata()

    # Convert the 3D image into 2D slices along the z-axis
    for z in range(image.shape[2]):
        slice = image[:, :, z]

        # Create a new NIfTI image from the slice
        slice_nii = nib.Nifti1Image(slice, nii.affine)

        # Save the slice as a NIfTI file
        slice_name = f"{nifti_file.stem}_slice_{z}.nii"
        slice_path = os.path.join(output_dir, slice_name)
        nib.save(slice_nii, slice_path)