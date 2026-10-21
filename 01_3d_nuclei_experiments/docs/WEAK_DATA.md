The provided `examples_segmented_results.zip` contains 7 large old segmentation
volumes (plus metadata), but NOT their corresponding nuclei intensity volumes.
Masks alone cannot train image -> segmentation/flow models.

To enable `pretrain_weak: true` in config.yaml, prepare matched PATCH FOLDERS
under `data/weak/<unique_patch_id>/`, each containing:
- nuclei_patch.tif: float32 already-preprocessed intensity image, (Z,Y,X)
- masks_patch.tif: int labels, (Z,Y,X), from the old segmentation
- flow_patch.npz: arr_0 shaped (3,Z,Y,X), center-diffusion flow
- profile.npy: original 8-vector intensity profile
- MetaData.json: Example_MetaData with source_specimen or source_image identifier

Pair masks and nuclei with the exact SAME coordinate space and transforms.
Do not use low-quality labels in validation. Never include a manually validated
sample in weak pretraining if it belongs to validation/test group.
Do not generate flow targets by a generic EDT gradient: that would change the
supervision target relative to the current center-diffusion approach.

Old results filenames/required raw TIFF names are indexed in old_labels_manifest.json.
