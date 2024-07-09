# App-Segmentation

## Description

This project is a tool to make segmentation of images. It uses the pre-trained model [SAM](https://github.com/facebookresearch/segment-anything) to make the segmentation of the images. At the application, there is a tool for manual segmentation, point segmentation and box segmentation. 

## Setup

To start the project, follow these steps:

1. Clone the repository: `git clone https://github.com/Lorenzuou/app-segmentation.git`
2. Navigate into the project directory: `cd app-segmentation`
3. write the command: 
```bash 
docker-compose  up -d
```

download the SAM weight sam_vit_h_4b8939.pth from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it in the folder `app-segmentation/sam-server`

## Usage

To use the application, follow these steps:

open the application on localhost:4002, this is the manual segmentation tool.

to use points segmentation, open the application on localhost:4002/sam 

to use box segmentation, open the application on localhost:4002/box

The instructions for each tool are in the respective pages.