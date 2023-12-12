![Golf Swing Analyzer Logo](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/logo.png)

# Golf Swing Analyzer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/Analyze_Video.ipynb)

Welcome to the **Golf Swing Analyzer** project! This innovative tool leverages advanced pose estimation models to analyze golf swings from video inputs. Our aim is to help golfers improve their technique by providing insightful feedback based on established golf theories and professional golfer statistics.

## Project Overview

The Golf Swing Analyzer processes a video of a golfer's swing and outputs three key images that highlight the main phases of the swing. Using a sophisticated pose estimation model, it identifies various angles and points in the golfer's posture and compares them with the ideal positions and movements prescribed in golf theory.

### Key Features

- **Pose Estimation**: Utilizes a state-of-the-art pose estimation model to accurately track the golfer's body positions throughout the swing.
- **Swing Phase Detection**: Automatically segments the golf swing into three main parts for detailed analysis.
- **Angle and Point Analysis**: Identifies critical angles and points in the golfer's swing and compares them to ideal standards.

### Technologies Used

- **MediaPipe**: This project uses the MediaPipe Pose Estimation model, a cutting-edge solution developed by Google for real-time, high-fidelity pose tracking. The model is capable of tracking 33 key points on the body, providing detailed and accurate data on body posture and movement. This technology is crucial for our analysis, allowing for precise measurement and evaluation of the golfer's swing against established golfing standards. 

### Results

- **Output Images**: The tool generates three images, each representing a significant phase of the golf swing. These images highlight key positions and angles colored (in red/green) based on correctness.
![Example Swing Analysis](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement/blob/main/output.png)

- **Analysis Report**: Alongside the images, a report is provided, explaining the alignment of the golfer's pose with ideal golfing standards. This includes a breakdown of incorrect positions, and offering valuable feedback for improvement.

 
 ```plaintext
Swing part TOP: 
-> WRONG: Left ankle, left hip and right shoulder angle should be approximately 180 degrees.
-> WRONG: Left arm should be approximately straight.

Swing part CONTACT: 
-> WRONG: Left arm should be approximately straight.
```

[Instructions on how to set up and run the project. Include any prerequisites, installation steps, and how to run the application.]

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/8Rx_0qAA)
*Reminder*
*   *Do not miss [deadline](https://su2.utia.cas.cz/labs.html#projects) for uploading your implementation and documentation of your final project.*
*   *Include working demo in Colab, in which you clearly demonstrate your results. Please refer to the Implementation section on the [labs website](https://su2.utia.cas.cz/labs.html#projects).*
*   *Do not upload huge datasets to GitHub repository.*

