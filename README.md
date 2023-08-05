# Introduction and Tutorial for CMap

# 1. Introduction 
CMap is a new segmentation computational pipeline for *C. elegans* time-lapse embryos. By explicitly incorporating the nucleus information from StarryNite to the result of cell membrane segmentation, CMap can segment
the *C. elegans* embryo at late 550-cell stage vs 350-cell stage by CShpaer.

## 1.1 Novel 3D visualized GUI data and software

* Windows version ITK-CVE-SNAP software link: https://portland-my.sharepoint.com/:u:/g/personal/zelinli6-c_my_cityu_edu_hk/EYqGjfoFp8NDjoRLdhEUBKMBPVytBpQNKNBqRN-MH_2F9g .
* Linux version ITK-CVE-SNAP software link: https://portland-my.sharepoint.com/:u:/g/personal/zelinli6-c_my_cityu_edu_hk/EUe1bduMu8RPi1MPNYRCxvoBxHz5l9KV-TRJFM7eAOV_1A?e=eOYnCZ .


# 2. Tutorial
We provide two ways to run CMap to segment and generate visualized 3D GUI data for your own time-lapse fluorescent images.

## 2.1 Run the Program Online via Google Colab
Google Colab is a computational online jupyter notebook which is provided by Google and designed for scientific studies. Please visit the website and have a brief understanding on https://colab.research.google.com/ . 

You don't need to know how to use python or jupyter notebook. If you are running the program on your own data, all you need to do is to prepare your tifR images and the corresponding CD{embryo_name}.csv files. Actually, the free google colab provides few computational resources. I strongly recommend you enlarge the number of processes dealing with the shape analysis to get the results with reasonable operation time. 

* Youtube Google Colab tutorial video for a quick start! https://youtu.be/lQyx5Z2wY90 ! 

### >>>>====Colab Running Steps====<<<<
* Download and upload or copy the Folder CMapCode to your google drive root directory. https://drive.google.com/drive/folders/1NWAxXAQuFf9sNafWkvGYNslUAQjAszbW?usp=sharing .
  * The folder tree is
    ```
    ├── Your google drive root
    │   ├── CMapCode
    |   │   ├── ckpts
    |   │   │   ├── CMap_model_epoch_50.pth
    |   │   ├── data
    |   │   │   ├── **.py
    |   │   ├── dataset
    |   |   │   ├── run
    |   |   |   │   ├── {embryo name}
    |   |   |   |   │   ├── tifR
    |   |   |   |   |   │   ├── {embryo name}_L1-t{time point}-p{slice number index}.tif
    |   |   │   ├── CDFiles
    |   |   |   │   ├── {embryo name}.csv
    |   |   │   ├── CellFate.xls
    |   |   │   ├── name_dictionary_cmap.csv
    |   |   │   ├── name_dictionary.csv
    |   |   │   ├── number_dictionary.csv
    |   |   │   ├── tissue_wise_name_dictionary.csv
    |   │   ├── experiment
    |   │   │   ├── TRAIN_TEST.yaml
    |   │   ├── models
    |   │   │   ├── *.py
    |   │   ├── utils
    |   │   │   ├── *.py
    ```
  * The example membrane image data is saved in the folder *your google drive root/CMapCode/dataset/run/{embryo name}/tifR/{embryo name}_L1-t{time point}-p{slice number index}.tif* and the CD file with nucleus location labeling is saved at *CMapCode/dataset/run/CDFiles/{embryo name}.csv*
  * Please upload your own data on your *your google drive root/CMapCode/dataset/run* following the folder structure
* Open the CMap colab jupyter, and save a copy to your own colab to run. https://colab.research.google.com/drive/13F2R7HzMra8_CWsaD3FYNnt-GdzD1Tpy?usp=sharing .
  * Follow the instruction, click and run the block one by one.
  * Download the Cell-wise and Fate-wise GUI data folders *your google drive root/CMapCode/GUIDataCellWise* and *your google drive root/CMapCode/GUIDataFateWise* . Open the folder with our ITK-CVE-SNAP!

## 2.2 Run the Program on Your PC (Linux) with Python Environment

If you are running this, you should know how to use python and jupyter notebook. But it is very easy and kind for beginners! You may spend less than one week to learn python and jupyter notebook, then you can start preprocessing, training, running, shape analyzing, and GUI visualization data generation!

* Youtube PC (linux) tutorial video for a quick start! https://youtu.be/h2-89Fr2CAQ ! 

* If you have your own 3D labeled data and going to train them, please generate and group the data as following folder structure. All these nii.gz files are composed of 2D slices and CD files. All the code could be found in the preprocessing part! You need to group the training data with your own python script because you need to label the SegCell with your own data.
  ```
  ├── Your project root directory (code, data, and temporary output)
  │   ├── ckpts
  |   │   │   ├── {network name}
  |   |   │   │   ├── ...
  │   ├── dataset
  |   │   ├── training
  |   |   │   ├── {embryo name}
  |   |   |   │   ├── PklFile
  |   |   |   |   │   ├── {embryo name}_{time point}.pkl
  |   |   |   │   ├── RawMemb
  |   |   |   |   │   ├── {embryo name}_{time point}_rawMemb.nii.gz
  |   |   |   │   ├── RawNuc
  |   |   |   |   │   ├── {embryo name}_{time point}_rawNuc.nii.gz
  |   |   |   │   ├── SegCell
  |   |   |   |   │   ├── {embryo name}_{time point}_segCell.nii.gz
  |   |   |   │   ├── SegMemb
  |   |   |   |   │   ├── {embryo name}_{time point}_segMemb.nii.gz
  |   |   |   │   ├── SegNuc
  |   |   |   |   │   ├── {embryo name}_{time point}_segNuc.nii.gz
  ...
  
  ```
* **If you are running the program on your own data, no training is needed**, all you need to do is to prepare your tifR images and the corresponding CD{embryo_name}.csv files.



### >>>>====Jupyter Notebook Running Steps====<<<<

The steps are explained in the jupyter notebook. You run it one by one and change some parameters then you could the results. I use 30 processes in the local tutorial which could be faster but using large memory, which helps in shape analysis step. Shape analyzing costs a lot of time, please be patient if you use a small number of processes.

* The root folder structure is as follows (complete version).
  ```
  ├── CMapCode
  │   ├── ckpts
  |   │   ├── TRAIN_TEST (Output of 2_Train.ipynb)
  |   |   │   ├── cfg.yaml
  |   |   │   ├── model_epoch_{epoch index}.pth
  │   │   ├── CMap_model_epoch_50.pth
  │   ├── data
  │   │   ├── **.py
  │   ├── dataset
  |   │   ├── training
  |   |   │   ├── ** (Input of 2_Train.ipynb)
  |   │   ├── run
  |   |   │   ├── {embryo name}
  |   |   |   │   ├── PklFile (Output of 1_DataProcess.ipynb)
  |   |   |   |   │   ├── {embryo name}_{time point}.pkl
  |   |   |   │   ├── RawMemb (Output of 1_DataProcess.ipynb)
  |   |   |   |   │   ├── {embryo name}_{time point}_rawMemb.nii.gz
  |   |   |   │   ├── SegCell (Output of 3_Run_segmentation.ipynb)
  |   |   |   |   │   ├── {embryo name}_{time point}_segCell.nii.gz
  |   |   |   │   ├── SegCellDivisionCells (Output of 3_Run_segmentation.ipynb)
  |   |   |   |   │   ├── {embryo name}_{time point}_segCell.nii.gz
  |   |   |   │   ├── SegMemb (Output of 3_Run_segmentation.ipynb)
  |   |   |   |   │   ├── {embryo name}_{time point}_segMemb.nii.gz
  |   |   |   │   ├── SegNuc (Output of 1_DataProcess.ipynb)
  |   |   |   |   │   ├── {embryo name}_{time point}-segNuc.nii.gz
  |   |   |   │   ├── tifR (Raw Input of 1_DataProcess.ipynb)
  |   |   |   |   │   ├── {embryo name}_L1-t{time point}-p{slice number index}.tif
  |   │   ├── CDFiles
  |   |   │   ├── {embryo name}.csv
  |   │   ├── CellFate.xls
  |   │   ├── name_dictionary_cmap.csv
  |   │   ├── name_dictionary.csv
  |   │   ├── number_dictionary.csv
  |   │   ├── tissue_wise_name_dictionary.csv
  |   │   ├── experiment
  |   │   │   ├── TRAIN_TEST.yaml
  |   │   ├── models
  |   │   │   ├── *.py
  |   │   ├── utils
  |   │   │   ├── *.py
  |   │   ├── logs (Output of 2_Train.ipynb or 3_Run_segmentation.ipynb)
  |   │   │   ├── TRAIN_TEST_train.txt 
  |   │   │   ├── TRAIN_TEST_run.txt
  │   ├── middle_output
  |   │   ├── {embryo name} (Output of 4_Run_cell_shape_analysis.ipynb)
  |   |   │   ├── {embryo name}_{time point}_segCell_contact.txt
  |   |   │   ├── {embryo name}_{time point}_segCell_surface.txt
  |   |   │   ├── {embryo name}_{time point}_segCell_volume.txt
  |   │   ├── GUIDataFateWise (Output of 5_GUIData_visualization.ipynb)
  |   |   │   ├── {embryo name}
  |   |   |   │   ├── **
  |   |   |   ├── name_dictionary.csv
  |   │   ├── GUIDataTissueWise (Output of 5_GUIData_visualization.ipynb)
  |   |   │   ├── **
  |   |   ├──NucLocFile (Output of 4_Run_cell_shape_analysis.ipynb)
  |   |   │   ├── {embryo name}
  |   |   |   │   ├── {embryo name}_{time point}_nucLoc.csv
  |   |   ├──statistics (Output of 4_Run_cell_shape_analysis.ipynb)
  |   |   │   ├── {embryo name}
  |   |   |   │   ├── {embryo name}_contact.csv
  |   |   |   │   ├── {embryo name}_surface.csv
  |   |   |   │   ├── {embryo name}_volume.csv
  │   ├── 1_DataProcess.ipynb
  │   ├── 2_Train.ipynb
  │   ├── 3_Run_cell_shape_analysis.ipynb
  │   ├── 4_Run_cell_shape_analysis.ipynb
  │   ├── 5_GUIData_visualization.ipynb
  │   ├── train.py
  │   ├── test.py
  │   ├── environment.yml
  ```

# Acknowledgement
* The validation process partly uses the code of *CShaper*.
* Some parts of this repository are referred to BraTS-DMFNet, e.g., the implementation of the model.

