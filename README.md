Reid by Salience Matching
=========================

Matlab code for our ICCV 2013 work "Person Re-identification by Salience Matching" 


Installation
------------
Download VIPeR dataset, and put the subfolders (\cam_a and \cam_b) into directory .\dataset\viper\
Download CUHK01 dataset, and put the dataset at directory .\dataset\campus

Demo
====
Currently, one demo is provided

- demo_salmatch_cuhk01.m : perform evaluation over CUHK01 dataset

Remark
======
- The approach will generate large amount of intermediate data in the cache directory, so make sure memory is large enough (at least 30GB)
- The training / testing partition is generated following the approach [SDALF](http://www.lorisbazzani.info/code-datasets/sdalf-descriptor/) 
- Parallel Toolbox can accellerate the computation, use matlabpool if necessary

Cite our work
=============
@inproceedings{zhao2013person,
 title = {Person Re-identification by Salience Matching},
 author={Zhao, Rui and Ouyang, Wanli and Wang, Xiaogang},
 booktitle = {IEEE International Conference on Computer Vision (ICCV)},
 year = {2013},
 month = {December},
 address = {Sydney, Australia}
}
