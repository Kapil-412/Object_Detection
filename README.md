# Project Title

A brief description of what this project does and who it's for

# Steps to run the project

1. Make a Virtual Environment for installation of modules and packages.
```
sudo apt install python3-venv 
python3 -m venv myenv
source myenv/bin/activate 
```
After this command myenv folder will be generated

2. After installing Virtual environment install modules from requirements.txt file
```
pip install -r requirements.txt
```

It will take time according to speed of internet.

3. After installing modules from requirements.txt change the path of your folder according to your location in file named `yolov8_config.yaml`

Change path of train and valid location.
`train location contains path of folder DATASET/train/images`
and
`valid location contains path of folder DATASET/valid/images`

4. After the changes done run the file yolo_train_script.py In this file i have set the `epochs to 10` if you have high end pc with gpu then you can set it between `50 to 100` for better accuracy of detection.
If you have high gpu pc remove batch = 8 for better model creation default is 16 it will run on 16 for better accuracy.
Also if you have gpu with more than 4 gb then set device = 'gpu' to run faster.
After all changes run the below command It will take time to make to model for detection.
```
python3 yolo_train_script.py
```

After successfull a run folder will generate that is the model we will be needing for the detection.

5. If you have dataset of images then you can run the file `detectionOnDataset.py` but first make a 2 folder with <folder_name>, one folder in which you will give the images you have as dataset for detection and other folder will be used to keep detection results
```
python3 detectionOnDataset.py
```
6. For live image input i have used IP Webcam which i have dowloaded from playstore available for android. Download it and click 3 dots on right side and click on start server, after that video camera will start, IP address will be mention below copy that IP address and placed it in the file name main.py line no 23 replaced it with <ip_address>. after replacing ip address run the file.
```
python3 main.py
```
