# Commands:
To train the video model:
1. separate_grasped_model_seg: for segmentation images
2. separate_grasped_model: for rgb images
   
```
python fitvid/scripts/train_fitvid.py --output_dir /home/arpit/test_projects/fitvid/run_test_seg --dataset_file /home/arpit/test_projects/OmniGibson/dynamics_model_dataset_seg/dataset.hdf5 --wandb_online
```

To train the grasped model:
1. separate_grasped_model_seg: to use segmentation images
2. separate_grasped_model: to use rgb images
```
python fitvid/scripts/train_grasped_model.py --output_dir run_seg_grasped --dataset_file /home/arpit/test_projects/OmniGibson/dynamics_model_dataset_seg/dataset.hdf5 --wandb_online
```

# FitVid Video Prediction Model

Implementation of [FitVid][website] video prediction model in JAX/Flax.

If you find this code useful, please cite it in your paper:
```
@article{babaeizadeh2021fitvid,
  title={FitVid: Overfitting in Pixel-Level Video Prediction},
  author= {Babaeizadeh, Mohammad and Saffar, Mohammad Taghi and Nair, Suraj 
  and Levine, Sergey and Finn, Chelsea and Erhan, Dumitru},
  journal={arXiv preprint arXiv:2106.13195},
  year={2020}
}
```

[website]: https://sites.google.com/view/fitvidpaper

## Method

FitVid is a new architecture for conditional variational video prediction. 
It has ~300 million parameters and can be trained with minimal training tricks.

![Architecture](https://i.imgur.com/ym8uOxB.png)

## Sample Videos

| Human3.6M             |  RoboNet |
:-------------------------:|:-------------------------:
![Humans1](https://i.imgur.com/y621cvE.gif)  |  ![RoboNet1](https://i.imgur.com/KsZDnh0.gif)
![Humans2](https://i.imgur.com/yMHkqoh.gif)  |  ![RoboNet2](https://i.imgur.com/fOYPNMx.gif)

For more samples please visit [FitVid][website].
[website]: https://sites.google.com/view/fitvidpaper

## Instructions

Get dependencies:

```sh
pip3 install --user tensorflow
pip3 install --user tensorflow_addons
pip3 install --user flax
pip3 install --user ffmpeg
```

Train on RoboNet:
```sh
python -m fitvid.train  --output_dir /tmp/output
```

Disclaimer: Not an official Google product.

