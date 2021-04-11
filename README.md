# ImageSegmentation
Image segmentation for glomerulii recognition on kidney tissue

Based on [this Kaggle Competition](https://www.kaggle.com/c/hubmap-kidney-segmentation)

[TransUNet](https://github.com/Beckschen/TransUNet)

## TODO
 - make preprocessing code that chunks big images and preserves masking
 - make pytorch dataset class
 - try pretrained TransUNet on test set
 - fine tune TransUNet by training on train set

## Setup
 - clone and cd into repo
 - `mkdir data/train`
 - `mkdir data/test`
 - add .json and .tiff files into train/ and test/ from Kaggle

## Contributors
 - Shaumik Ashraf
 - Henry Wu
 - Allan Bishop

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
