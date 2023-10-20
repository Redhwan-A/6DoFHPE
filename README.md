# 6DoFHPE
**Paper titled: Real-time 6DoF Full-Range Markerless Head Pose Estimation**





<p align="center">
  <img src="https://github.com/Redhwan-A/6DoF-HPE/blob/main/gif/6DoF-Hpe.gif" alt="animated" />
</p>




# Datasets

* **CMU Panoptic**  from [here](http://domedb.perception.cs.cmu.edu/).



## **Our results**

| Method         | Retrain? | Yaw   | Pitch | Roll  | MAE   |
|----------------|----------|-------|-------|-------|-------|
| WHENet         | No       | 37.96 | 22.7  | 16.54 | 25.73 |
| HopeNet        | No       | 20.40 | 17.47 | 13.40 | 17.09 |
| FSA-Net        | No       | 17.52 | 16.31 | 13.02 | 15.62 |
| Img2Pose       | No       | 12.99 | 16.69 | 15.64 | 15.11 |
| DirectMHP      | Yes      | 5.86  | 8.25  | 7.25  | 7.12  |
| DirectMHP      | No       | 5.75  | 8.01  | 6.96  | 6.91  |
| 6DRepNet       | Yes      | 5.20  | 7.22  | 6.00  | 6.14  |
| 6DoF-HPE (ours)  | Yes      | **5.13**  | **6.99**  | **5.77**  | **5.96** |
| WHENet         | No       | 29.87 | 19.88 | 14.66 | 21.47 |
| DirectMHP      | Yes      | 7.38  | 8.56  | 7.47  | 7.80  |
| DirectMHP      | No       | 7.32  | 8.54  | 7.35  | 7.74  |
| 6DRepNet       | Yes      | 5.89  | 7.76  | 6.39  | 6.68  |
| 6DoF-HPE (ours)  | Yes      | **5.83**  | **7.63**  | **6.35**  | **6.60**  |


## **run code**

To run **demo_6DoF.py**. You need to install ROS from [here](https://wiki.ros.org/Distributions).

The roscore can be launched using the roscore executable:

```
roscore
```
Then run an RGB-D camera, for example, if it is  realsense d435i. 

```
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud,colorizer align_depth:=true ordered_pc:=true
```

Download the pre-trained RepVGG model '**cmu.pth**'  for the full range angles or '**300W_LP.pth**' for narrow range angles from [here](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq) and pre-trained SSD model from [here](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq)  and then save them in their certain directory (please, see our code to know their paths).



# Citing

Coming soon!
