# SCREAM: SCene REndering Adversarial Model for Non-overlapping Registration and Ground Generation
## Requirements
```
# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
``` 
## Pretrained params and Datasets
We provide
* pretrained params on 3DMatch, KITTI, OpenGG  
Download [params (168.92MB)](https://pan.baidu.com/s/1DM_wOAsAQ8eAPmflZzpuNg) with code "8qnj".
* processed 3DMatch (include 3DMatch_train, 3DMatch_val, 3DMatch_test, 3DLoMatch_test and 3DZeroMatch_test. Each point cloud pair is downsampled with a voxel size 0.065. Both the training and validation sets contain point cloud pairs with zero overlap rate)  
Download [3DMatch (2.42GB)](https://pan.baidu.com/s/1v3u8TevCq7FGspNkO-t9yw) with code "f51d".
* processed KITTI (include KITTI_train, KITTI_val, KITTI_test. Each point cloud pair is downsampled with a voxel size 0.7)  
Download [KITTI (380.31MB)](https://pan.baidu.com/s/1wzBb9_FSOFk4XWIABemqTA) with code "8hff".
* OpenGG (include OpenGF_train, OpenGF_val, OpenGF_test, which contains DSM-DEM pairs)  
Download [OpenGG (992.66MB)](https://pan.baidu.com/s/1_OkeH-UHbvxWbkBMJPzBkg) with code "3kvh"

In addition, you can also download the 3DMatch provided in [PREDATOR](https://github.com/prs-eth/OverlapPredator) and use ThreeDMatchDataset_PREDATOR in "datasets/three_d_match.py" parses it.  
you can also download [raw KITTI (26.76GB)](https://pan.baidu.com/s/1Ig-prVTaeb6BL9l7Ab-UVg) with code "w5kz", and use KITTI_PREDATOR in "datasets/kitti.py" parses it. But this is not necessary.


## Registration on 3DMatch
### visualize
Please run ```visualize_3d_match.py```, you can visualize registration on 3DLoMatch or 3DzeroMatch by calling ```visualize_3dmatch(lo_loader)``` and ```visualize_zero_match()```.
```
if __name__ == '__main__':
    visualize_3dmatch(lo_loader)
    # visualize_zero_match()
```
### test
Please call ```evaluate_3d_match```/```evaluate_3d_lo_match```/```evaluate_3d_zero_match``` in ```evaluate_3d_match.py``` to evaluate the model's performance on 3DMatch/3DLoMatch/3DZeroMatch.
### train
Please run ```train_3d_match.py```

## Registration on KITTI
### visualize
Please run ```visualize_kitti.py```
### test
Please run ```evaluate_kitti.py```
### train
Please run ```train_kitti.py```

## Ground Generation on OnenGG
### visualize
Please run ```visualize_open_gf.py```
### test
Please run ```evaluate_open_gf.py```
### train
Please run ```train_open_gf.py```

## The comparison methods in the paper
We express our sincere thanks to the authors of these methods
* [PerfectMatch](https://github.com/zgojcic/3DSmoothNet)
* [FCGF](https://github.com/chrischoy/FCGF)
* [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
* [PREDATOR](https://github.com/prs-eth/OverlapPredator)
* [SpinNet](https://github.com/QingyongHu/SpinNet)
* [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
* [REGTR](https://github.com/yewzijian/RegTR)
* [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)