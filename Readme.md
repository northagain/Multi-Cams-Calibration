# Calibrate Multi Cameras

## Deployment

To deploy this project run

You should first set the numbers of cameras and images in calibrate_stereo.py line 60 to 62.  
Make sure your file directory is like **File Directory**
```shell
  python calibrate_stereo.py
```


## Environment Variables

To run this project, you will need to install the following packages:  

numpy  
opencv
## File Directory
Your file directory should be like this:  
**test** is for 1d-calibration pole images  
**calibrate/i** is for chess board images with a second folder  
**validation** is for L pole images
```
│  calibrate_initial.py
│  calibrate_stereo.py
│  match_points.py
│  match_points_L.py
│  validation.py
│  Readme.md
│  
├─test
│      L1-01.bmp
│       ...
│      L2-01.bmp
│       ...
│      L3-01.bmp
│       ...
├─validation
│      L1-01.bmp
│       ...
│      L2-01.bmp
│       ...
│      L3-01.bmp
│       ...    
├─calibrate
│  ├─1
│  │      L1-01.bmp
│  │        ...
│  │      
│  ├─2
│  │      L2-01.bmp
│  │        ...
│  │      
│  ├─3
│  │      L3-01.bmp
└──│    ...
}
```

### For Validation
Make sure you already got the output of cameras in folder Result
1. Set the numbers of cameras and images in match_points_L.py line 233 to 236.  
2. Run match_points_L.py, you will find 1.txt,2.txt,...,n.txt created in the validation folder.
3. Set the numbers of cameras in validation.py line 60.
4. Run validation.py, you will see the visual and scalar result.
