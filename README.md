# Team Epoch forestbiomass

This repository contains all code created by Team Epoch during participation in the forestbiosmass challenge.

The forestbiomass challenge asked participants to use satelite imagery to estimate biomass levels of Finnish forests. These satelite images where made by the Sentinel-1 and Sentinel-2 ESA satelites. To enable models to learn how to use these images, ground truth was provided in the form of AGBM Lidar images. Although these images provide a more accurate estimation of biomass levels, their acquisition requires flying an aircraft across the region being measured. Being able to convert the satelite images into good approximations of Lidar measurements allows for faster and more sustainble monitoring of biomass.

The repository contains the following directories:

- analysis contains files and scripts used during EDA.
- computingpower contains files necessary for usage of the Delft-Blue cluster.
- data contains files related to processing of the competition dataset.
- models contains all the different model types tried during the competition.
- utils contains utility scripts.
