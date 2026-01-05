# Semantic-Guided Diffusion Models for Enhanced Image Restoration
Image restoration, a pivotal challenge in computer vision, aims to recover high-fidelity visuals from degraded inputs. Traditional methods often falter under complex degradations due to a lack of high-level semantic reasoning. Addressing this, we introduce SR-Diff, a diffusion-based framework that integrates semantic priors from vision-language models (VLMs) to guide the restoration process. By refining semantic embeddings through a Semantic Refinement Module, SR-Diff ensures robust guidance with minimal computational overhead. Experiments on dehazing, deraining, motion deblurring, and low-light enhancement demonstrate SR-Diff's superiority over state-of-the-art methods, achieving significant improvements in both quantitative metrics and perceptual realism. Here, we show that semantic guidance enhances restoration fidelity, particularly in complex scenes, marking a substantial advancement in image restoration technology.

[Qiang Zhu],
[ZhiBo Wang],
[MengYao Wang],
[JiaXing Hu],
[YuGuo Wu]

## Some Results
![workplace]()
<div align='center'>Comparison of LPIPS and SSIM performance under different mask ratios using vairous methods</div>
![workplace]()
<div align='center'>Comparison of LPIPS and SSIM performance under different mask ratios using vairous methods</div>
![workplace]()
<div align='center'>Comparison of LPIPS and SSIM performance under different mask ratios using vairous methods</div>
![workplace]()
<div align='center'>Comparison of LPIPS and SSIM performance under different mask ratios using vairous methods</div>

## How to run the code?

### Dependencies
* nvidia:
  - cuda: 11.4
* python 3.8
### Install
pip install -r requirements.txt

### Dataset Preparation
Preparing the train and test datasets following our paper Dataset Construction section as:

```bash
#### for training dataset ####
#### (uncompleted means inpainting) ####
datasets/train
|--hazy
|  |--LQ/*.png
|  |--GT/*.png
|  |--LQCLS/*.pt
|  |--RestoreCLS/*.pt
|--low-light
|--rainy
|--motion-blurry
```



#### Dataset Links
| Degradation |                                  motion-blurry                                  |                                                 hazy                                                 |                                                  rain                                                 |                                   low-light                                   |             
|-------------|:-------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
| Datasets    | [Gopro](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view) | [RESIDE-6k](https://drive.google.com/drive/folders/1XVD0x74vKQ0-cqazACUZnjUOWURXIeqH?usp=drive_link) | [Rain100H](http://www.icst.pku.edu.cn/struct/att/RainTrainH.zip) | [LOL](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view) | 
| RestoreCLS  | [Gopro](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view) | [RESIDE-6k](https://drive.google.com/drive/folders/1XVD0x74vKQ0-cqazACUZnjUOWURXIeqH?usp=drive_link) | [Rain100H](http://www.icst.pku.edu.cn/struct/att/RainTrainH.zip) | [LOL](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view) |

#### for testing dataset ####
#### (the same structure as train) ####
datasets/val
...


