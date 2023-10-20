# HPE
Frm-Hpe: Full-Range Markerless Head Pose Estimation

# Preparing datasets
Download datasets:

* **300W-LP**, **AFLW2000** from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).

* **BIWI** (Biwi Kinect Head Pose Database) from [here](https://icu.ee.ethz.ch/research/datsets.html)

Store them in the *datasets* directory.

For 300W-LP and AFLW2000 we need to create a *filenamelist*.
```
python create_filename_list.py --root_dir datasets/300W_LP
```




| Method         | Retrain? |   Yaw  |  Pitch |  Roll  |   MAE  |
|----------------|----------|--------|--------|--------|--------|
| WHENet         |   No     | 37.96  |  22.7  | 16.54  | 25.73  |
| HopeNet        |   No     | 20.40  | 17.47  | 13.40  | 17.09  |
| FSA-Net        |   No     | 17.52  | 16.31  | 13.02  | 15.62  |
| DirectMHP (Yes)|  Yes    | 5.86   | 8.25   | 7.25   | 7.12   |
| DirectMHP (No) |  No     | 5.75   | 8.01   | 6.96   | 6.91   |
| 6DRepNet (Yes) |  Yes    | 5.20   | 7.22   | 6.00   | 6.14   |
| Frm-Hpe (ours) |  Yes    | **5.13**   | **6.99**   | **5.77**   | **5.96**   |
| WHENet         |   No     | 29.87  | 19.88  | 14.66  | 21.47  |
| DirectMHP      |  Yes     | 7.38   | 8.56   | 7.47   | 7.80   |
| DirectMHP      |  No     | 7.32   | 8.54   | 7.35   | 7.74   |
| 6DRepNet       |  Yes    | 5.89   | 7.76   | 6.39   | 6.68   |
| Frm-Hpe (ours) |  Yes    | **5.83**   | **7.63**   | **6.35**   | **6.60** |








# Citing

Coming soon!
