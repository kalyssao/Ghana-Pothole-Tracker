# Ghana Pothole Tracker
This is the online repository for the Ghana Pothole Tracker project. The code written runs on a Raspberry Pi and uses image classification methods to determine the presence of potholes on a road in realtime, and transmits the co-ordinates of said potholes to a MySQL database.
These co-ordinates are then plotted on a web application to inform user's route selection. Additionally, they are available for download.

The purpose of this project is to introduce an element of crowdsourcing for civil good in user journeys, making it easier for the mobile maintenance units (MMU's) to gather data about road improvement projects.

### Demo
![App Screenshot](https://imgur.com/a/0IDhGfk)

### Dataset
Expected input data is a folder of images, with the naming convention of the folder contents representing the image classes. eg. **pothole.1.png** & **not_a_pothole.1.png** <br/>
The full stops are necessary, as the script splits the filenames in order to get the image label.

### Usage
These files are meant to be run via command line, with the following syntax: 

**python3 file_name.py --dataset /path/to/dataset/here**<br/>

The classification metrics and confusion matrix for the given data are the outputs.

An extra script (test.py) is included to allow the user to verify the results with their own input images.
