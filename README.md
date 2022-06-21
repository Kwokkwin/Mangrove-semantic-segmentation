# Mangrove-semantic-segmentation
 
**Project Title** : Mangrove and Associated Community Classification from Historical Remote Sensing Data

**Project Description**: Moreton Bay has rich mangroves and associated saltmarsh communities over 18,500 ha of intertidal areas. They provide essential support for biodiversity, fisheries and blue carbon sink. Continuous monitoring and a better understanding of their communities are essential for their protection and management of current and future rising sea levels. <img width="415" alt="Screen Shot 2022-06-22 at 8 50 30 am" src="https://user-images.githubusercontent.com/70581958/174909572-258b4a0e-13ca-468b-b3b2-7093c6616229.png">


During the past forty years, the State of Queensland has undertaken continuous studies mapping and assessing the change of mangrove and saltmarsh communities. The traditional study used remote sensing imagery for manual extraction of mangrove regions, which is time-consuming and subjective. 

This project aims to develop deep learning based methods to automatically detect different mangrove and saltmarsh communities and species from high-resolution airborne imagery. Historical maps and images will be used as the labels and data to train deep neural network models, and then validated on the most recent maps. Students who are interested in this project shall have basic knowledge about image processing and machine learning, and good Python programming skills. A final report and a demo of the classification system are expected as the output of the project.

**Project pre-analysis**
veg1_color.xlsx: Vegetation mask color chosen(43 class)

**Project Structure**

<img width="469" alt="Screen Shot 2022-06-22 at 8 51 26 am" src="https://user-images.githubusercontent.com/70581958/174909665-6fe13945-99aa-46cb-8aaf-bdf7d905f5cc.png">
arcgis_dataset: raw data extracted from ArcGIS( 1m resolution 2500*2500*3 RGB images and 10cm resolution 250*250 grey vegs(labels))
img_dataset: all "*.image.tif" files from arcgis_dataset
veg_dataset: all "*.veg.tif" files from arcgis_dataset

**Data Clean**
dataset_filter.ipynb:  

balance_class.ipynb:
split_pic.ipynb: Split 2500*2500 image to 256*256 image
veg_resize.ipynb: Expanise each pixel by 10*10
morphology.ipynb: Erode the non-smooth boundaries.(to be considered)
separate_dataset(622).ipynb
