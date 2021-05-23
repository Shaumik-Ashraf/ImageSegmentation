# ImageSegmentation
Image segmentation for glomerulii recognition on kidney tissue

Based on [this Kaggle Competition](https://www.kaggle.com/c/hubmap-kidney-segmentation)

Inspiration from [TransUNet](https://github.com/Beckschen/TransUNet)

## MobileTransUNet
We just hacked MobileNet into the TransUNet hybrid, replacing ResNet. 
MobileTransUNet achieved a Dice Score of 0.8015 on the Kaggle's 
public test set **without** pretraining, which is pretty decent, but 
not SOTA. See code in TransUNet submodule. 

## Contributors
 - Shaumik Ashraf
 - Henry Wu
 - Allan Bishop

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
