import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import cv2
import csv
import os
import math
import datetime
import numpy as np

class Yolov7_PoseEstimation:
    def __init__(self, file_path, csv_file_path, output_video_path):
        self.file_path = file_path
        self.csv_file_path = csv_file_path
        self.output_video_path = output_video_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):
        model = torch.load('yolov7-w6-pose.pt', map_location=self.device)['model']
        model.float().eval()

        if torch.cuda.is_available():
            model.half().to(self.device)
        return model

    def calculate_angle(self, a_x, a_y, b_x, b_y, c_x, c_y):
        radians = math.atan2(c_y - b_y, c_x - b_x) - math.atan2(a_y - b_y, a_x - b_x)
        angle = math.degrees(radians)
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

    def calculate_angle2(self, x1, y1, x2, y2, axis='x', orientation='right'):
      if (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * x1) != 0:
          if axis == 'x':
              theta = math.acos((x2 - x1) * (-x1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * x1))
          elif axis == 'y':
              theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
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

    def middle_point(self, a_x, a_y, b_x, b_y):
        midpoint_x = (a_x + b_x) / 2
        midpoint_y = (a_y + b_y) / 2
        return midpoint_x, midpoint_y

    def run_inference(self, image):
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)

        if torch.cuda.is_available():
            image = image.half().to(self.device)

        image = image.unsqueeze(0)

        with torch.no_grad():
            output, _ = self.model(image)

        return output, image

    def save_keypoints(self, kpts, steps):
        num_kpts = len(kpts) // steps
        coords = []

        for kid in range(num_kpts):
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]

            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        coords.append(0)
                        coords.append(0)
                        continue

            coords.append(x_coord)
            coords.append(y_coord)

        return coords

    def draw_keypoints(self, output, image):
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)

        with torch.no_grad():
            output = output_to_keypoint(output)

        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        all_coords = []

        for idx in range(output.shape[0]):
            coords = self.save_keypoints(output[idx, 7:].T, 3)
            all_coords.append(coords)

        return nimg, all_coords

    def write_csv_file(self, coordinates, a):
        scale_factor = 0.68

        # Determine the correct file path
        csv_file = self.csv_file_path if a == 0 else self.csv_file_path.replace('.csv', f'_{a+1}.csv')

        # Check if file exists to write headers
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
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

            scaled_coordinates = coordinates[:6] + [int(coord * scale_factor) for coord in coordinates[6:]]
            writer.writerow(scaled_coordinates)

    def pose_estimation(self):
        cap = cv2.VideoCapture(self.file_path)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame_number += 1

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, frame = self.run_inference(frame)
                frame, all_coords = self.draw_keypoints(output, frame)
                
                a = 0
                for person in all_coords:
                    # Get coordinates of specific landmarks
                    left_shoulder_x = person[10]
                    left_shoulder_y = person[11]
                    right_shoulder_x = person[12]
                    right_shoulder_y = person[13]
                    left_hip_x = person[22]
                    left_hip_y = person[23]
                    right_hip_x = person[24]
                    right_hip_y = person[25]
                    left_wrist_x = person[18]
                    left_wrist_y = person[19]
                    right_wrist_x = person[20]
                    right_wrist_y = person[21]
                    nose_x = person[0]
                    nose_y = person[1]
                    left_knee_x = person[26]
                    left_knee_y = person[27]
                    right_knee_x = person[28]
                    right_knee_y = person[29]
                    left_ankle_x = person[30]
                    left_ankle_y = person[31]
                    right_ankle_x = person[32]
                    right_ankle_y = person[33]
                    left_elbow_x = person[14]
                    left_elbow_y = person[15]

                    # Calculate angles
                    knee_angle = self.calculate_angle(left_hip_x, left_hip_y, left_knee_x, left_knee_y, left_ankle_x, left_ankle_y)
                    pelvis_angle = self.calculate_angle(left_ankle_x, left_ankle_y, left_hip_x, left_hip_y, right_shoulder_x, right_shoulder_y)
                    arm_angle = self.calculate_angle(left_wrist_x, left_wrist_y, left_elbow_x, left_elbow_y, left_shoulder_x, left_shoulder_y)
                    shoulders_inclination = self.calculate_angle2(right_shoulder_x, right_shoulder_y,
                                                            left_shoulder_x, left_shoulder_y, 'x', 'left')
                    hips_inclination = self.calculate_angle2(left_hip_x, left_hip_y, right_hip_x, right_hip_y, 'x')
                    midpoint_x, midpoint_y = self.middle_point(right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y)
                    # Get the timestamp
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    video_timestamp = round(frame_number / fps)
                    video_timestamp = str(datetime.timedelta(seconds=video_timestamp))


                    # Display points
                    cv2.circle(frame, (int(right_shoulder_x), int(right_shoulder_y)), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (int(left_shoulder_x), int(left_shoulder_y)), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (int(right_hip_x), int(right_hip_y)), 6, (255, 255, 0), -1)
                    cv2.circle(frame, (int(left_hip_x), int(left_hip_y)), 6, (0, 150, 255), -1)
                    cv2.circle(frame, (int(right_knee_x), int(right_knee_y)), 6, (255, 0, 255), -1)
                    cv2.circle(frame, (int(left_knee_x), int(left_knee_y)), 6, (255, 0, 255), -1)
                    cv2.circle(frame, (int(left_ankle_x), int(left_ankle_y)), 6, (255, 0, 0), -1)
                    cv2.circle(frame, (int(left_wrist_x), int(left_wrist_y)), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (int(nose_x), int(nose_y)), 6, (0, 0, 255), -1)
                    cv2.circle(frame, (int(left_elbow_x), int(left_elbow_y)), 6, (128, 0, 128), -1)
                    cv2.circle(frame, (int(right_ankle_x), int(right_ankle_y)), 6, (255, 0, 0), -1)
                    cv2.circle(frame, (int(midpoint_x), int(midpoint_y)), 6, (255, 255, 255), -1)

                    # Display angle and lines on the image
                    cv2.line(frame, (int(left_ankle_x), int(left_ankle_y)), (int(left_ankle_x), int(left_ankle_y)- 200), (255, 0, 0), 2)
                    cv2.line(frame, (int(right_ankle_x), int(right_ankle_y)), (int(right_ankle_x), int(right_ankle_y) - 200), (255, 0, 0), 2)

                    cv2.line(frame, (int(right_shoulder_x), int(right_shoulder_y)), (int(right_shoulder_x) + 100, int(right_shoulder_y)), (0, 255, 0), 2)
                    cv2.line(frame, (int(left_shoulder_x), int(left_shoulder_y)), (int(right_shoulder_x), int(right_shoulder_y)), (0, 255, 0), 2)
                    cv2.line(frame, (int(left_hip_x), int(left_hip_y)), (int(left_hip_x) - 100, int(left_hip_y) ), (255, 255, 0), 2)
                    cv2.line(frame, (int(left_hip_x), int(left_hip_y)), (int(right_hip_x), int(right_hip_y)), (255, 255, 0), 2)

                    cv2.line(frame, (int(left_hip_x), int(left_hip_y)), (int(left_knee_x), int(left_knee_y)), (255, 0, 255), 2)
                    cv2.line(frame, (int(left_knee_x), int(left_knee_y)), (int(left_ankle_x), int(left_ankle_y)), (255, 0, 255), 2)

                    cv2.line(frame, (int(left_hip_x), int(left_hip_y)), (int(left_ankle_x), int(left_ankle_y)), (0, 150, 255), 2)
                    cv2.line(frame, (int(left_hip_x), int(left_hip_y)), (int(right_shoulder_x), int(right_shoulder_y)), (0, 150, 255), 2)

                    cv2.line(frame, (int(left_shoulder_x), int(left_shoulder_y)), (int(left_elbow_x), int(left_elbow_y)), (128, 0, 128), 2)
                    cv2.line(frame, (int(left_elbow_x), int(left_elbow_y)), (int(left_wrist_x), int(left_wrist_y)), (128, 0, 128), 2)
                    cv2.line(frame, (int(midpoint_x), int(midpoint_y)), (int(midpoint_x), int(midpoint_y) - 200), (255, 255, 255), 2)

                    # Write data to CSV
                    self.write_csv_file([video_timestamp, shoulders_inclination, hips_inclination,
                                        knee_angle, pelvis_angle, arm_angle,
                                        int(right_shoulder_x), int(right_shoulder_y),
                                        int(left_shoulder_x), int(left_shoulder_y),
                                        int(left_elbow_x), int(left_elbow_y),
                                        int(right_wrist_x), int(right_wrist_y),
                                        int(left_wrist_x), int(left_wrist_y),
                                        int(nose_x), int(nose_y),
                                        int(right_hip_x), int(right_hip_y),
                                        int(left_hip_x), int(left_hip_y),
                                        int(right_knee_x), int(right_knee_y),
                                        int(left_knee_x), int(left_knee_y),
                                        int(right_ankle_x), int(right_ankle_y),
                                        int(left_ankle_x), int(left_ankle_y),
                                        int(midpoint_x), int(midpoint_y)], a)

                    a += 1

                if a == 1:
                    cv2.putText(frame, f'Shoulders inclination: {shoulders_inclination:.2f}', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Hips inclination: {hips_inclination:.2f}', (10, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f'Knee Angle: {knee_angle:.2f}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(frame, f'Pelvis Angle: {pelvis_angle:.2f}', (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
                    cv2.putText(frame, f'Arm Angle: {arm_angle:.2f}', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

                frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
                out.write(frame)
            else:
                break

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()