# End-2-end-self-supervised-SLAM
Project for ETH 3D vision course

# Downloaders

TUM:
bash script for downloading TUM (to add more download files update the raw_data_downloader_TUM.sh file)
```shell
bash sample_data/dataset_TUM/raw_data_downloader_TUM.sh 
```
Running pointfusion file with the correct relative string to the data
```shell
python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss "depth_consistency
```

NYU:
downloader doesnt work, need download by hand:
https://onedrive.live.com/?authkey=%21AKVvEAT14TgiFEE&cid=36712431A95E7A25&id=36712431A95E7A25%212551&parId=36712431A95E7A25%21472&action=locate


# Setup

Install gradslam by navigating to "gradslam/" and executing "pip install ."

Execute "pip install -e ." in the main folder


# Folder structure

The main code of our project should be in the "end2endslam" directory
"gradslam" contains a copy of the gradslam repository.
"perception/monodepth2" contains the modified monodepth2 repository



