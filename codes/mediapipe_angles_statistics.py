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
#file_name = 0
file_name = 'videos_160/00.mp4'


# Save the file under the name with date-time ending
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_file_path = f'variables_{timestamp}.csv'

#set up media pipe- create two variables
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    # Calculate the angle between three points in degrees
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    #radians = math.atan2(c[2] - b[2], c[1] - b[1]) - math.atan2(a[2] - b[2], a[1] - b[1])
    angle = math.degrees(radians)
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle


# find angle between two vectors
# first vector joining two points (x1, y1) and (x2, y2)
# second vector joining point (x1, y1) and any point on the y-axis passing through (x1, y1), in our case we set it at (x1, 0)
def calculate_angle2(x1, y1, x2, y2, axis = 'x', orientation ='right'):
    
    if (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * x1) != 0: 
        if axis == 'x':
            theta = math.acos((x2 - x1) * (-x1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * x1))   
        elif axis == 'y':
            theta = math.acos( (y2 -y1)*(-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1))
        else:
            raise ValueError("Invalid axis, use 'x' or 'y'")
            
            
        if orientation == 'right':
            angle = int(180/math.pi) * theta
        elif orientation == 'left':
            angle = 180 - int(180/math.pi) * theta
        else:
            raise ValueError("Invalid orientation, use 'left' or 'right'")
            
    else:
        return 0  # Handle the case where x1 is zero to avoid division by zero 
    
    return angle


def middle_point(a, b):
    midpoint_x = (a.x + b.x) / 2
    midpoint_y = (a.y + b.y) / 2
    return midpoint_x, midpoint_y


with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    if view == 'front':
        writer.writerow(["video_timestamp", "shoulders_inclination", "hips_inclination", 
                         "knee_angle", "pelvis_angle", "arm_angle",
                         "left_shoulder X", "left_shoulder Y", 
                         "right_shoulder X", "right_shoulder Y", 
                         "left_elbow X", "left_elbow Y",
                         "left_wrist X", "left_wrist Y", 
                         "nose X", "nose Y", 
                         "left_hip X", "left_hip Y",
                         "right_knee X", "right_knee Y",
                         "right_ankle X", "right_ankle Y"
                         "left_ankle X", "left_ankle Y",
                         "midpoint X", "midpoint Y"])
    elif view == 'side':
        writer.writerow([])
                
if file_name == 0:
    cap = cv2.VideoCapture(file_name)
else:
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'{file_name}_out_{timestamp}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_number = 0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_timestamp = round(frame_number / fps)
            video_timestamp = str(datetime.timedelta(seconds=video_timestamp))
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
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            
            
            if view == 'front':
                
                # Calculate angles
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                pelvis_angle = calculate_angle(left_ankle, left_hip, right_shoulder)
                arm_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
                shoulders_inclination = calculate_angle2(int(right_shoulder.x * w), int(right_shoulder.y * h), int(left_shoulder.x * w), int(left_shoulder.y * h), 'x', 'left')
                hips_inclination = calculate_angle2(int(left_hip.x * w), int(left_hip.y * h), int(right_hip.x * w), int(right_hip.y * h), 'x')
                
                midpoint_x, midpoint_y = middle_point(right_ankle, left_ankle)
                
                # Open the CSV file in write mode
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data to the CSV file
                    writer.writerow([video_timestamp, shoulders_inclination, hips_inclination, 
                                     knee_angle, pelvis_angle, arm_angle,
                                     int(left_shoulder.x* w), int(left_shoulder.y* h), 
                                     int(right_shoulder.x* w), int(right_shoulder.y* h),
                                     int(left_elbow.x * w), int(left_elbow.y * h),
                                     int(left_wrist.x* w), int(left_wrist.y* h), 
                                     int(nose.x* w), int(nose.y* h), 
                                     int(left_hip.x * w), int(left_hip.y * h),
                                     int(right_knee.x * w), int(right_knee.y * h),
                                     int(right_ankle.x * w), int(right_ankle.y * h),
                                     int(left_ankle.x * w), int(left_ankle.y * h),
                                     int(midpoint_x * w), int(midpoint_y * h)])
                    
                # Display points
                cv2.circle(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(image, (int(right_hip.x * w), int(right_hip.y * h)), 6, (255, 255, 0), -1)
                cv2.circle(image, (int(left_hip.x * w), int(left_hip.y * h)), 6, (0, 150, 255), -1)
                cv2.circle(image, (int(right_knee.x * w), int(right_knee.y * h)), 6, (255, 0, 255), -1)
                cv2.circle(image, (int(left_knee.x * w), int(left_knee.y * h)), 6, (255, 0, 255), -1)
                cv2.circle(image, (int(left_ankle.x * w), int(left_ankle.y * h)), 6, (255, 0, 0), -1)
                cv2.circle(image, (int(left_wrist.x * w), int(left_wrist.y * h)), 6, (0, 255, 255), -1)
                cv2.circle(image, (int(nose.x * w), int(nose.y * h)), 6, (0, 0, 255), -1)
                cv2.circle(image, (int(left_elbow.x * w), int(left_elbow.y * h)), 6, (128, 0, 128), -1)
                cv2.circle(image, (int(right_ankle.x * w), int(right_ankle.y * h)), 6, (255, 0, 0), -1)
                cv2.circle(image, (int(midpoint_x * w), int(midpoint_y * h)), 6, (255, 255, 255), -1)
                
                # Display angle and lines on the image
                cv2.putText(image, f'Shoulders inclination: {shoulders_inclination:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.line(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), (int(right_shoulder.x * w) + 100, int(right_shoulder.y * h)), (0, 255, 0), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 255, 0), 2)
                
                cv2.putText(image, f'Hips inclination: {hips_inclination:.2f}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_hip.x * w) - 100, int(left_hip.y * h) ), (255, 255, 0), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(right_hip.x * w), int(right_hip.y * h)), (255, 255, 0), 2) 
                
                cv2.putText(image, f'Knee Angle: {knee_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_knee.x * w), int(left_knee.y * h)), (255, 0, 255), 2)
                cv2.line(image, (int(left_knee.x * w), int(left_knee.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)), (255, 0, 255), 2)
                
                cv2.putText(image, f'Pelvis Angle: {pelvis_angle:.2f}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)), (0, 150, 255), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 150, 255), 2)
                
                cv2.putText(image, f'Arm Angle: {arm_angle:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(left_elbow.x * w), int(left_elbow.y * h)), (128, 0, 128), 2)
                cv2.line(image, (int(left_elbow.x * w), int(left_elbow.y * h)), (int(left_wrist.x * w), int(left_wrist.y * h)), (128, 0, 128), 2)
                
                cv2.line(image, (int(left_ankle.x * w), int(left_ankle.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)- 200), (255, 0, 0), 2)
                cv2.line(image, (int(right_ankle.x * w), int(right_ankle.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)), (255, 255, 255), 2)
            
            elif view == 'side': 
                print()
                
            else:
                print("You have just two options: 'front' or 'side'")
                break
            frame_number += 1
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