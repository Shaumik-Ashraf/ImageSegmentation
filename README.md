# ImageSegmentation
Image segmentation for glomerulii recognition on kidney tissue

Based on [this Kaggle Competition](https://www.kaggle.com/c/hubmap-kidney-segmentation)

## File Spec
 - `data/` contains part of the data from kaggle competition in same structure
   + you should create a `data/train/` folder that contains a *.json and *.tiff 
 - `feature_extractor.py` extracts images
 - `truth_extractor.py` extracts the FTU segments from JSON files
 - `visualizer.py` displays an image
 - `rle_encoder.py` writes the submission file

## Data format
 - in `data/train/` (you must add these files:)
   + `xxxxxxxxx.json` contains the glomerulii polygon coordinates
   + `xxxxxxxxx-anatomical-structures.json` contains the medulla polygon coordinates
   + `xxxxxxxxx.tiff` contains the image

**The `train.csv` file contains the same data as the .json file**

## Contributors
 - Shaumik Ashraf
 - Henry Wu
 - Allan Bishop

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
