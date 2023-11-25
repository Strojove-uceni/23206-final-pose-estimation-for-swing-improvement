import cv2
import mediapipe as mp
import math
import csv
import datetime
import os

def estimate(fl, slow_motion_factor=1):
    view = 'front'
    file_name = f'cropped_videos/{fl}.mp4'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_path = f'results_both_wrists/variables_{fl}.csv'

    # Create the results directory if it doesn't exist
    if not os.path.exists(os.path.dirname(csv_file_path)):
        os.makedirs(os.path.dirname(csv_file_path))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Additional processing for slow motion effect
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if view == 'front':
            writer.writerow(["timestamp", 'right_wrist_x', 'right_wrist_y','left_wrist_x','left_wrist_y'])
        elif view == 'side':
            writer.writerow([])

    cap = cv2.VideoCapture(file_name)
    fps = cap.get(cv2.CAP_PROP_FPS) / slow_motion_factor  # Adjust FPS for slow motion
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f'results_both_wrists/{fl}_out_{timestamp}.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    frame_number = 0 
    # fps=30
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # Process each frame slow_motion_factor times
            for _ in range(slow_motion_factor):
                duplicated_image = image.copy()

                try:
                    h, w = duplicated_image.shape[:2]

                    duplicated_image = cv2.cvtColor(duplicated_image, cv2.COLOR_BGR2RGB)
                    duplicated_image.flags.writeable = False

                    keypoints = pose.process(duplicated_image)

                    duplicated_image.flags.writeable = True
                    duplicated_image = cv2.cvtColor(duplicated_image, cv2.COLOR_RGB2BGR)

                    landmarks = keypoints.pose_landmarks.landmark
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]


                    if view == 'front':
                        # Adjust timestamp for slow motion
                        video_timestamp = round(frame_number / fps / slow_motion_factor, 2)
                        with open(csv_file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([video_timestamp, right_wrist.x * w, right_wrist.y * h,left_wrist.x * w, left_wrist.y * h])

                        cv2.circle(duplicated_image, (int(right_wrist.x * w), int(right_wrist.y * h)), 6, (255, 0, 255), -1)
                        cv2.circle(duplicated_image, (int(left_wrist.x * w), int(left_wrist.y * h)), 6, (255, 0, 255), -1)

                    elif view == 'side': 
                        pass
                    else:
                        print("Only 'front' or 'side' views are supported")
                        break

                except AttributeError:
                    pass

                # Write the frame with the drawn circle to the output video
                out.write(duplicated_image)

                # cv2.imshow('Mediapipe Pose', duplicated_image)

                if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty('Mediapipe Pose', cv2.WND_PROP_VISIBLE) < 1:
                    break

            frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return fps

