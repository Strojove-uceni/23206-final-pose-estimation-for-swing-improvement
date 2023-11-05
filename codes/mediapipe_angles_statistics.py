# Install MediaPipe Python package, write to cmd
# pip install mediapipe

import cv2 #opencv
import mediapipe as mp
import math
import csv
import datetime

# Set the view: 'side' or 'front'
view = 'front'

# If file_name = 0 it will show real-time webcam source
# otherwise you can write the path to your video file
file_name = 0
#file_name = 'videos_160/11.mp4'


# Save the file under the name with date-time ending
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_file_path = f'variables_{timestamp}.csv'

#set up media pipe- create two variables
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# find angle between two vectors
# first vector joining two points (x1, y1) and (x2, y2)
# second vector joining point (x1, y1) and any point on the y-axis passing through (x1, y1), in our case we set it at (x1, 0)
def calculate_angle(x1, y1, x2, y2, orientation='right'):
    theta = math.acos( (x2 -x1)*(-x1) / (math.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * x1) )
    if orientation == 'right':
        angle = int(180/math.pi)*theta
    elif orientation == 'left':
        angle = 180 - int(180/math.pi)*theta
    else:
        raise ValueError("Invalid orientation, use 'left' or 'right'")
    return angle

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    if view == 'front':
        writer.writerow(["shoulders_inclination", "hips_inclination"])
    elif view == 'side':
        writer.writerow([])
                
if file_name == 0:
    cap = cv2.VideoCapture(file_name)
else:
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'{file_name}_out_{timestamp}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = image.shape[:2]
    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
    
            keypoints = pose.process(image)
    
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
            landmarks = keypoints.pose_landmarks.landmark
            
            # Get coordinates of specific landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            
            if view == 'front':
                
                # Calculate angles
                shoulders_inclination = calculate_angle(int(right_shoulder.x * w), int(right_shoulder.y * h), int(left_shoulder.x * w), int(left_shoulder.y * h), 'left')
                hips_inclination = calculate_angle(int(left_hip.x * w), int(left_hip.y * h), int(right_hip.x * w), int(right_hip.y * h))
                
                # Open the CSV file in write mode
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data to the CSV file
                    writer.writerow([shoulders_inclination, hips_inclination])
                    
                # Display points
                cv2.circle(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(image, (int(right_hip.x * w), int(right_hip.y * h)), 6, (255, 255, 0), -1)
                cv2.circle(image, (int(left_hip.x * w), int(left_hip.y * h)), 6, (255, 255, 0), -1)
                
                # Display angle and lines on the image
                cv2.putText(image, f'Shoulders inclination: {shoulders_inclination:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), (int(right_shoulder.x * w) + 100, int(right_shoulder.y * h)), (0, 255, 0), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 255, 0), 2)
                
                cv2.putText(image, f'Hips inclination: {hips_inclination:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_hip.x * w) - 100, int(left_hip.y * h) ), (255, 255, 0), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(right_hip.x * w), int(right_hip.y * h)), (255, 255, 0), 2) 
            
            elif view == 'side': 
                print()
                
            else:
                print("You have just two options: 'front' or 'side'")
                break
        except AttributeError:
            pass
        

        if file_name == 0:
            cv2.imshow('Mediapipe Pose', image)
        else:
            imS = cv2.resize(image, (640, 360))
            cv2.imshow('Mediapipe Pose', imS)
            out.write(image)

        if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty('Mediapipe Pose', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
if file_name != 0:
    out.release()
cv2.destroyAllWindows()