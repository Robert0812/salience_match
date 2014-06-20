Person Reid by Salience Matching
=========================

Matlab code for our ICCV 2013 work "Person Re-identification by Salience Matching" 


Installation
------------
- Download VIPeR dataset, and put the subfolders (\cam_a and \cam_b) into directory .\dataset\viper\
- Download CUHK01 dataset, and put the dataset at directory .\dataset\campus

Demo
----
Currently, one demo is provided

- demo_salmatch_cuhk01.m : perform evaluation over CUHK01 dataset

Remark
------
- The approach will generate large amount of intermediate data in the cache directory
- The training / testing partition is generated following the approach [SDALF](http://www.lorisbazzani.info/code-datasets/sdalf-descriptor/) 
- Parallel Toolbox can accellerate the computation, use matlabpool if necessary

Cite our work
-------------
Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Person Re-identification by Salience Matching. In ICCV 2013.
Rui Zhao, Wanli OUyang, and Xiaogang Wang. Unsupervised Salience Learning for Person Re-identification. In CVPR 2013.
