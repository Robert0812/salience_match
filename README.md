Person Reid by Salience Matching
=========================

Matlab code for our ICCV 2013 work "Person Re-identification by Salience Matching".  

Created by [Rui Zhao](http://www.ee.cuhk.edu.hk/~rzhao/), June, 2014


##Installation
- Download VIPeR dataset, and put the subfolders (/cam_a and /cam_b) into directory .\dataset\viper\
- Download CUHK01 dataset, and put the dataset at directory ./dataset/campus
- If you run on Windows system, you need to replace the '/' with '\' in following script files, set_paths.m, ./code/normdata.m, ./code/initialcontext_general.m, ./demo_salmatch_cuhk01.m

##Demo
Currently, one demo is provided

- demo_salmatch_cuhk01.m : perform evaluation over CUHK01 dataset

##Remark
- Running the demo_salmatch_cuhk01.m is supposed to achieve 30.04% at rank-1 accuracy on CUHK01 dataset
- The training / testing partition is generated following the approach [SDALF](http://www.lorisbazzani.info/code-datasets/sdalf-descriptor/) 
- Parallel Toolbox can accellerate the computation, use matlabpool if necessary
- The demo will generate large amount of intermediate data in the cache directory (around 25.5GB for CUHK01 demo)
- This demo was tested on MATLAB (R2012b), both 64-bit Win7 and 64-bit Linux with Intel Xeon 3.33 GHz CPU.

##Citing our works
Please kindly cite our work in your publications if it helps your research:
    
    @inproceedings{zhao13person,
      Author = {Zhao, Rui and Ouyang, Wanli and Wang, Xiaogang},
      Title = { Person Re-identification by Salience Matching},
      booktitle = {IEEE International Conference on Computer Vision (ICCV)},
 	  year = {2013}
    }

And our [previous work](https://github.com/Robert0812/salience_reid) published in CVPR 2013:

    @inproceedings{zhao13unsupervised,
      Author = {Zhao, Rui and Ouyang, Wanli and Wang, Xiaogang},
      Title = { Unsupervised Salience Learning for Person Re-identification },
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year = {2013}
    }

<!-- - Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Person Re-identification by Salience Matching. In ICCV 2013.
- Rui Zhao, Wanli OUyang, and Xiaogang Wang. Unsupervised Salience Learning for Person Re-identification. In CVPR 2013. -->
