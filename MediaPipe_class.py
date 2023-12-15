import cv2
import mediapipe as mp
import math
import csv
import datetime

class MediaPipe_PoseEstimation:
    # Initializes the MediaPipe_PoseEstimation class
    def __init__(self, input_file, csv_file_name, output_video_name):
        self.input_file = input_file
        self.csv_file_name = csv_file_name
        self.output_video_name = output_video_name
        
    # Calculates the angle between three points using the arctangent method
    def calculate_angle(self, a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

    # Calculates the angle between two points and one of the axes
    def calculate_angle2(self, x1, y1, x2, y2, axis='x', orientation='right'):
        if (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * x1) != 0:
            if axis == 'x':
                theta = math.acos((x2 - x1) * (-x1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * x1))
            elif axis == 'y':
                theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
            else:
                raise ValueError("Invalid axis, use 'x' or 'y'")

            if orientation == 'right':
                angle = int(180 / math.pi) * theta
            elif orientation == 'left':
                angle = 180 - int(180 / math.pi) * theta
            else:
                raise ValueError("Invalid orientation, use 'left' or 'right'")
        else:
            return 0

        return angle

    # Calculates the midpoint between two points
    def middle_point(self, a, b):
        midpoint_x = (a.x + b.x) / 2
        midpoint_y = (a.y + b.y) / 2
        return midpoint_x, midpoint_y

    # Processes the input video, calculates pose statistics, and generates an output video and write the statistics into .csv file
    def process_video(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with open(self.csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["video_timestamp", "shoulders_inclination", "hips_inclination",
                             "knee_angle", "pelvis_angle", "arm_angle",
                             "right_shoulder X", "right_shoulder Y",
                             "left_shoulder X", "left_shoulder Y",
                             "left_elbow X", "left_elbow Y",
                             "right_wrist X", "right_wrist Y",
                             "left_wrist X", "left_wrist Y",
                             "nose X", "nose Y",
                             "right_hip X", "right_hip Y",
                             "left_hip X", "left_hip Y",
                             "right_knee X", "right_knee Y",
                             "left_knee X", "left_knee Y",
                             "right_ankle X", "right_ankle Y",
                             "left_ankle X", "left_ankle Y",
                             "midpoint X", "midpoint Y"])

        cap = cv2.VideoCapture(self.input_file)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.output_video_name, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

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

                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]

                    knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                    pelvis_angle = self.calculate_angle(left_ankle, left_hip, right_shoulder)
                    arm_angle = self.calculate_angle(left_wrist, left_elbow, left_shoulder)
                    shoulders_inclination = self.calculate_angle2(int(right_shoulder.x * w), int(right_shoulder.y * h),
                                                                  int(left_shoulder.x * w), int(left_shoulder.y * h),
                                                                  'x', 'left')
                    hips_inclination = self.calculate_angle2(int(left_hip.x * w), int(left_hip.y * h),
                                                              int(right_hip.x * w), int(right_hip.y * h), 'x')

                    midpoint_x, midpoint_y = self.middle_point(right_ankle, left_ankle)

                    with open(self.csv_file_name, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([video_timestamp, shoulders_inclination, hips_inclination,
                                         knee_angle, pelvis_angle, arm_angle,
                                         int(right_shoulder.x * w), int(right_shoulder.y * h),
                                         int(left_shoulder.x * w), int(left_shoulder.y * h),
                                         int(left_elbow.x * w), int(left_elbow.y * h),
                                         int(right_wrist.x * w), int(right_wrist.y * h),
                                         int(left_wrist.x * w), int(left_wrist.y * h),
                                         int(nose.x * w), int(nose.y * h),
                                         int(right_hip.x * w), int(right_hip.y * h),
                                         int(left_hip.x * w), int(left_hip.y * h),
                                         int(right_knee.x * w), int(right_knee.y * h),
                                         int(left_knee.x * w), int(left_knee.y * h),
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
                    cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_hip.x * w) - 100, int(left_hip.y * h)), (255, 255, 0), 2)
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

                    cv2.line(image, (int(left_ankle.x * w), int(left_ankle.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h) - 200), (255, 0, 0), 2)
                    cv2.line(image, (int(right_ankle.x * w), int(right_ankle.y * h)), (int(right_ankle.x * w), int(right_ankle.y * h) - 200), (255, 0, 0), 2)

                    cv2.line(image, (int(midpoint_x * w), int(midpoint_y * h)), (int(midpoint_x * w), int(midpoint_y * h) - 200), (255, 255, 255), 2)

                    # Write the frame into the file
                    out.write(image)

                    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit the video window
                        break

                    frame_number += 1

                except Exception as e:
                    print(f"An error occurred: {e}")

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()