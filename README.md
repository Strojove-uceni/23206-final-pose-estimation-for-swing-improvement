![Golf Swing Analyzer Logo](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/logo.png)

# Golf Swing Analyzer
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/Analyze_Video.ipynb)

Authors: *Soňa Drocárová*, *Tereza Fucsiková*

Welcome to the **Golf Swing Analyzer** project! This innovative tool leverages advanced pose estimation models to analyze golf swings from video inputs. Our aim is to help golfers improve their technique by providing insightful feedback based on established golf theories and professional golfer statistics.

## Project Overview

The Golf Swing Analyzer processes a video of a golfer's swing and outputs three key images that highlight the main phases of the swing. Using a sophisticated pose estimation model, it identifies various angles and points in the golfer's posture and compares them with the ideal positions and movements prescribed in golf theory.

### Key Features

- **Pose Estimation**: Utilizes a state-of-the-art pose estimation model to track the golfer's body positions throughout the swing accurately.
- **Swing Phase Detection**: Automatically segments the golf swing into three main parts for detailed analysis.
- **Angle and Point Analysis**: Identifies critical angles and points in the golfer's swing and compares them to ideal standards.

### Technologies Used

- **MediaPipe**: This project uses the MediaPipe Pose Estimation model, a cutting-edge solution developed by Google for real-time, high-fidelity pose tracking. The model is capable of tracking 33 key points on the body, providing detailed and accurate data on body posture and movement. This technology is crucial for our analysis, allowing for precise measurement and evaluation of the golfer's swing against established golfing standards.

- **YOLO (You Only Look Once) for Pose Estimation**: Another option for pose estimation that is implemented in our project is YOLO. This approach should be a fast and accurate method for pose estimation. However, during our trials, we found that while YOLO provided promising results, MediaPipe offered more precise and detailed pose data, which was crucial for our specific application in analyzing the intricacies of golf swings.

Since there is no data available for pose estimation evaluation on golf swing videos, the accuracy of our approach implemented with both of these models is evaluated on 173 YouTube videos of professional swings (list of these was found in [1] and based on it videos were scraped and cropped in length). Therefore, we assumed that these professional swings should produce as few mistakes as possible and the number of errors for each pose estimator is visualized in the following bar graphs.

![Results Parts](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/errors_comparison_mistakes.png)
![Results Whole](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/errors_comparison_plot.png)

### How It Works
The Golf Swing Analyzer follows a structured process to analyze and provide feedback on golf swings. A step-by-step overview of the code execution process is provided in the following diagram. 
![Process](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/diagram.png)

1. **Video Input**: The process begins with uploading a video of a golf swing.
2. **Applying MediaPipe Pose Estimation**: Once the video is received, it's processed using the MediaPipe Pose Estimation model. This model identifies and tracks key points on the golfer's body throughout the swing. In addition to preserving the information about body keypoints at each frame, different body angles such as the arm angle in the elbow or the angle in the knee are calculated and tracked.
3. **Swing Phase Segmentation**: The code then segments the swing into distinct phases (e.g., address, top backswing, contact) for targeted analysis. The input to this process is the information about every body part and angle that is examined at every frame of the video. In golf swing, what comes naturally to both more and less advanced players is the arm movement trajectory. Based on this trajectory, the extreme values are detected which are then used to find the frames of interest and discard the rest. An example of this is shown in the image where the swing parts are labelled along with the wrist function extrema.
![Swing Phase Segmentation](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/swing_split_example.jpg)
5. **Comparison with Golf Theory**: The calculated angles and points are compared against established standards in golf theory to determine their correctness.
6. **Generating Output**: This process generates a set of images capturing the key moments of the swing, along with an analysis report. The report highlights areas of the swing that deviate from the ideal golfing technique.


### Results

- **Output Images**: The tool generates three images, each representing a significant phase of the golf swing. These images highlight key positions and angles colored (in red/green) based on correctness.
![Example Swing Analysis](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/output.png)

- **Analysis Report**: Alongside the images, a report is provided, explaining the alignment of the golfer's pose with ideal golfing standards. This includes a breakdown of incorrect positions, and offering valuable feedback for improvement.

 
 ```plaintext
Swing part TOP: 
-> WRONG: Left ankle, left hip and right shoulder angle should form a straight line. Try turning more into the backswing.
-> WRONG: The left arm should be straight at this point.

Swing part CONTACT: 
-> WRONG: The left arm should be straight at this point.
```
### Conclusion
The Golf Swing Analyzer project successfully demonstrates the potential of advanced pose estimation technologies, like MediaPipe, in analyzing and improving sports techniques. The implications of this project extend beyond golf training. The methodology and technology applied here could be adapted for other sports where technique and posture play a crucial role, such as tennis or baseball.

While our project has yielded promising results, it's important to acknowledge certain limitations. While MediaPipe provides high-precision tracking, there's room for improvement in pose estimation accuracy when dealing with sports poses and sudden movements. Enhancing the model's robustness to handle these scenarios will make the analysis more reliable. Currently, the golf swing is segmented into a few key phases. Further breaking down the swing into more phases could provide deeper insights into the mechanics of the swing, allowing for more detailed feedback and analysis. 

To sum up, while our system has some limitations when analyzing swings from professionals recorded during tournaments, it's important to recognize that these conditions are often less ideal than the typical environment in which an amateur golfer might record their practice sessions. The controlled angles and surroundings of golfers recording themselves for this purpose provide clearer data, whereas videos recorded in tournament settings present more variability such as more people moving around in the background, less ideal angle for recording or the player being positioned further from the camera than ideal.


### References
1. **GolfDB: A Video Database for Golf Swing Sequencing**  
   McNally, William et al. (2019). "GolfDB: A Video Database for Golf Swing Sequencing." In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. June 2019.  
   [Paper Link](https://github.com/wmcnally/golfdb)

2. **MediaPipe: A Framework for Perceiving and Processing Reality**  
   Lugaresi, Camillo et al. (2019). "MediaPipe: A Framework for Perceiving and Processing Reality." In: Third Workshop on Computer Vision for AR/VR at IEEE Computer Vision and Pattern Recognition (CVPR) 2019.  
   [Paper Link](https://mixedreality.cs.cornell.edu/s/NewTitle_May1_MediaPipe_CVPR_CV4ARVR_Workshop_2019.pdf)

3. **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors**  
   Wang, Chien-Yao et al. (2023). "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors." In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.  
   [Repo Link](https://github.com/WongKinYiu/yolov7?fbclid=IwAR1mtD0Vq5tlIPLxnPshr4CkKSNkpwOx4yWXuTxRjGctCGIAU6-hdPQI6pM)


