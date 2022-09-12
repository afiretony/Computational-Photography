# Developing RAW Images
## Requirements
```
skimage
numpy
matplotlib
scipy
```
## Description
`RAW_develop.py` contains functions to converting `RAW` images to `.png` including linearization, white-balancing, demosaicing, color space correction, brightness adjustment and gamma encoding.


## Example
```
python im_processing.py --filename ../data/campus.tiff --AWB preset --brightness 0.25 --pattern rggb --show_image
```