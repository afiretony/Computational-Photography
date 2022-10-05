# HDR imaging

## Develop from RAW

The following command was used for developing RAW images took with a Nikon camera:

```shell
dcraw -q 3 -T -4 -f -w -o 1 <input image name>.NEF
```

```
-q 3 : Use Adaptive Homogeneity-Directed (AHD) interpolation.
-T   : Write TIFF with metadata instead of PDM/PPM/PAM.
-4   : Linear 16-bit.
-f   : Interpolate RGB as four colors.
-w   : Use the white balance specified by the camera.
-o 1 : sRGB D65 (default) color space.
```

## HDR Merging

Merge JPG images to get HDR image, optionally, you can set these flags:

```
python HDR_MERGE_JPG.py 
      --scene <path-to-JPG-dir> 
      --weight_type <"uniform", "tent", "gaussian", "photon", "optimal">
      --downsample <downsampling rate>
      --z_min <Z min to clip>
      --z_max <Z max to clip>
      --merge_mode <"linear" or "log">
```

Unlike RAW images which are linear in intrinsic, JPG images are non-linear and additionally needed to be linearized.

Merge RAW images to get HDR image, optionally, you can set these flags:

```
python HDR_MERGE_RAW.PY
      --scene <path-to-JPG-dir> 
      --weight_type <"uniform", "tent", "gaussian", "photon", "optimal">
      --downsample <downsampling rate>
      --z_min <Z min to clip>
      --z_max <Z max to clip>
      --merge_mode <"linear" or "log">
```

## Color Correction and White balancing

Color correction:

```
python color_correction.py
```

White balancing:

```
python white_balancing.py
```

## Tone Mapping

Tone mapping:

```
python tone_mapping.py
```

## Noise Calibration and Estimating Optimal Weight

Noise Calibration:

```
python noise_calibration.py
```

