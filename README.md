# DeepStream Human Pose Estimation

Human pose estimation is the computer vision task of estimating the configuration (‘the pose’) of the human body by localizing certain key points on a body within a video or a photo. The following application serves as a reference to deploy custom pose estimation models with DeepStream 5.0 using the [TRTPose](https://github.com/NVIDIA-AI-IOT/trt_pose) project as an example. 

A detailed deep-dive NVIDIA Developer blog is available [here](https://developer.nvidia.com/blog/creating-a-human-pose-estimation-application-with-deepstream-sdk/?ncid=so-link-52952-vt24&sfdcid=EM08#cid=em08_so-link_en-us).
<!--<img src="images/input.gif" width="300"/> <img src="images/auxillary.png" width="100"/> <img src="images/output.gif" width="300"/>-->

<table>
  <tr>
    <td>Input Video Source</td>
     <td></td>
     <td>Output Video</td>
  </tr>
  <tr>
    <td valign="top"><img src="images/input.gif"></td>
    <td valign="center"><img src="images/auxillary.png" width="100"></td>
    <td valign="top"><img src="images/output.gif"></td>
  </tr>
 </table>


## Prerequisites
You will need 
1. DeepStreamSDK 5.x
2. CUDA 10.2
3. TensorRT 7.x


## Getting Started:

### Using Docker (Jetson Only)

**Build a container**
~~~
git clone https://github.com/MACNICA-CLAVIS-NV/deepstream_pose_estimation
~~~
~~~
cd deepstream_pose_estimation
~~~
~~~
chmod +x *.sh
~~~
~~~
./docker_build.sh
~~~

**Run the application**
**You need to have a USB camera on /dev/video0 on your host L4T OS.**
~~~
./docker_run.sh
~~~

**Note: This release supports only for JetPack 4.5.1.**  
If you want to run this on other versions of JetPack, modify the following line in Dockerfile to select the base image which support your version. Refer to [the DeepStream-l4t repository page in NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:deepstream-l4t/tags) to find the right base image.
~~~
ARG BASE_IMAGE=nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples
~~~

### Normal Install
To get started, please follow these steps.
1. Install [DeepStream](https://developer.nvidia.com/deepstream-sdk) on your platform, verify it is working by running deepstream-app.
2. Clone the repository in your directory.
2. Download the TRTPose [model](https://github.com/NVIDIA-AI-IOT/trt_pose), convert it to ONNX using this [export utility](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/trt_pose/utils/export_for_isaac.py), and set its location in the DeepStream configuration file.  
Or you can use the **pose_estimation.onnx** in this repository.
5. Compile the program
 ```
  $ cd deepstream-pose-estimation/
  $ make
  $ ./deepstream-pose-estimation-app <file-uri> <output-path>
```
5. The final output is rendered to the display with X11/EGL and is also stored in 'output-path' as `Pose_Estimation.mp4`. 
6. You can input image from V4L2 camera like USB web cam.
```
  $ ./deepstream-pose-estimation-app <camera device>
```
Here is a example:
```
  $ ./deepstream-pose-estimation-app /dev/video0
```

NOTE: If you do not already have a .trt engine generated from the ONNX model you provided to DeepStream, an engine will be created on the first run of the application. Depending upon the system you’re using, this may take anywhere from 4 to 10 minutes.

For any issues or questions, please feel free to make a new post on the [DeepStreamSDK forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/).

## References
Cao, Zhe, et al. "Realtime multi-person 2d pose estimation using part affinity fields." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple baselines for human pose estimation and tracking." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
