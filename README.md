# Introduction and Tutorial for CMap

# 1. Introduction 
CMap is a new segmentation computational pipeline for *C. elegans* time-lapse embryos. By explicitly incorporating the nucleus information from StarryNite to the result of cell membrane segmentation, CMap can segment
the *C. elegans* embryo at late 550-cell stage vs 350-cell stage by CShpaer.

## 1.1 Novel 3D visualized GUI data and software


# 2. Tutorial
We provide two ways to run CMap to segment and generate visualized 3D GUI data for your own time-lapse fluorescent images.

## 2.1 Run the Program Online via Google Colab
Google Colab is a computational online jupyter notebook which is provided by Google and designed for scientific studies. Please visit the website and have a brief understanding on https://colab.research.google.com/ . 

You don't need to know how to use python or jupyter notebook. If you are running the program on your own data, all you need to do is preparing your tifR images and the corresponding CD{embryo_name}.csv files. Actually, the free colab provide few computational resource. I strongly 

* Youtube tutorial video for a quick start! https://youtu.be/lQyx5Z2wY90 ! 

REF THE TIFF IMAGE ON GOOGLE DRIVE
REF THE CD FILE ON GOOGLE DRIVE

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
    |   │   │   ├── *.py
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

## 2.2 Run the Program on Your PC with Python Environment

You need to know how to use python or jupyter notebook. 

* If you are running the program on your own data, no training is needed, all you need to do is preparing your tifR images and the corresponding CD{embryo_name}.csv files.
* If you have your own 3D labeled data and going to train them, please generate and group the data as following folder structure:


* Youtube tutorial video for a quick start! 



### >>>>====Jupyter Notebook Running Steps====<<<<


# Acknowledgement

* Some parts of this repository are referred to BraTS-DMFNet, e.g., the implementation of the model.

