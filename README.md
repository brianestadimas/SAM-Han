# SAM_Robustness

SAM_Robustness_Evulate includes two evualtions

## style transfer

## corruption

# Style transfer

How to run the code?

1.  Download models
    Download [decoder.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing)/[vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under models/.

2.  The dataset we used for style transfer

    #### content image:

          https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/filled-silhouettes

    #### style image:

         https://github.com/BathVisArtData/PeopleArt

3.  Generate the images after style transfer

```
  bash style_transfer.sh
```

4. Generated the mask for the stylized images

```
python calculate_stylized_mask.py
```

5. Evualate the mIoU between the clean images and the stylized transferred images

```
python calculate_mIoU.py
```

# Common corruption

How to run

1. Prepare dataset

```
data/
|____clean_images/
|    |__img1.txt
|    |__img2.txt
|    |__ ...
|____perturbed_images/
     |__img1.txt
     |__img2.txt
     |__ ...
```

# Notes

1. Generate Feature Maps with sam_han.py
2. Run train and test with sam_han_multi_train.py
