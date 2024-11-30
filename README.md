<p align="center">
    <img width="800" src="./assets/teaser.pdf".
</p>

# SGDreamer: Scene Harmonic Gaussian-based Dreamer for Domain-Free Text-to-3D Generation with Peripheral 360-Degree Views


<div align="center">
<!-- [![arXiv](https://img.shields.io/badge/ArXiv-2310.11784-b31b1b.svg?logo=arXiv)]() -->
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ZhenyuSun-Walker/DreamGaussian/blob/main/LICENSE) 

</div>

> #### SGDreamer: Scene Harmonic Gaussian-based Dreamer for Domain-Free Text-to-3D Generation with Peripheral 360-Degree Views
> ##### [Zhenyu Sun](https://zhenyusun-walker.github.io/), [Xiaohan Zhang](https://github.com/Xiaohan-Z/), [Qi Liu](https://drliuqi.github.io/), [Huan Wang](https://huanwang.tech/)
######

## 🔨 Installation
### Clone the repository

  ```
  git clone https://github.com/ZhenyuSun-Walker/SGDreamer.git
  cd SGDreamer
  ```

### Create the conda environment and install the dependencies:

  ```bash
  conda create -n SGDreamer python=3.10.14
  conda activate SGDreamer
  pip install -r requirements.txt
  pip install MVRec/submodules/diff-gaussian-rasterization
  pip install MVRec/submodules/simple-knn
  ```

## Runnig the code
### Multi-view panorama generation:
```bash
cd MVGen # make sure you are under the SGDreamer/MVGen

# Default Example 
python generate.py --gen_video --save_frames [Other options]
python select_range.py --source ./outputs/$results --target ../generate_mvimages
```
#### Other Options
- `--fov` : Denote the horizontal field of camera view, 90 in degrees as default.
- `--deg` : Specify the rotation angle around the vertical axis, 45 in degrees as default.
- `--prompt_folder` : Path to the text file containing the prompts including different scenes.  

Specifically, you neee to download the weight file from [here](https://pan.baidu.com/s/18M39ZzGIuyNTFZYJ7BfItQ?pwd=3Z8X), and then put it under 
```MVGen/weights/pano/last/```.

To train the pano-generation model, please download data from [matterport3D](https://niessner.github.io/Matterport/) skybox data and [labels](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0).

To use your own data, please also follow the organization as follows: 
```
├── SGDreamer
    ├── MVRec
      ├── data
          ├── mp3d_skybox
            ├── train.npy
            ├── test.npy
            ├── 5q7pvUzZiYa
              ├──blip3
              ├──matterport_skybox_images
            ├── 1LXtFkjw3qL
            ├── ....
```


Now the project structure is shown as be below:
 ```
  SGDreamer
    ├── MVRec  
    ├── MVGen                   
        ├── generate.py                
        ├── select_range.py
        ├── outputs   
            ├── <results_1>
        ├── weights
    ├── generate_mvimages
        ├── <results_1>
            ├── <scene_1>
                ├── images
    ...
 ```

### FlowMap for Camera Calculation
 ```bash
  cd ../flowmap
  CUDA_VISIBLE_DEVICES=1 python3 -m flowmap.overfit dataset=images dataset.images.root=../generate_mvimages/$results/$scene/images
 ```

The checkpoint used to initialize FlowMap can be downloaded [here](https://drive.google.com/drive/folders/1PqByQSfzyLjfdZZDwn6RXIECso7WB9IY?usp=drive_link). 
Organize the weight file under ```SGDreamer/flowmap/checkpoints```.

### 3D Scene Reconstruction
```bash
cd ../MVRec
CUDA_VISIBLE_DEVICES=0 python train.py -s ../flowmap/outputs/local/colmap/ --name $scene
```
Then you can check the rendered images, metrics and the gaussian pointclouds in 
```MVRec/rendered_images  MVRec/metrics  MVRec/output``` respectively.

## 🤗 Acknowledgement

We deeply appreciate [MVDiffusion](https://github.com/Tangshitao/MVDiffusion), [FlowMap](https://github.com/dcharatan/flowmap), [Gaussian_Barf](https://github.com/cameronosmith/gaussian_barf) and [MVGS](https://github.com/xiaobiaodu/MVGS) for their models.