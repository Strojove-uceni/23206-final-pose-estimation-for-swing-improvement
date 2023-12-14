import cv2 #opencv
import pandas as pd
import os
import numpy as np
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt

class DataProcessor:
    """
    Handles the loading and processing of the dataset.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = None

    def load_data(self):
        # Keep only the golfer from detected peaple
        all_csvs= [file for file in os.listdir(self.folder_path) if file.endswith('.csv')] # video and csv has to be in same file (one folder for each video and csv file)
        # It is usually the first person but to be sure we pick the person standing the widest
        filename=all_csvs[0].split('.')[0]
        if len(all_csvs)>1: 
                ankle_width=[]
                for i in range(len(all_csvs)):
                    data_current=pd.read_csv(self.folder_path + all_csvs[i])
                    ankle_width.append(data_current["left_ankle X"].iloc[0]-data_current["right_ankle X"].iloc[0])
                golfer_id=pd.Series(ankle_width).idxmax()
                if golfer_id>0:
                    filename=filename+'_'+str(golfer_id+1)

        self.data = pd.read_csv(self.folder_path+'/'+filename+'.csv')

    def split_swing(self):
        """
        Keeps only 3 key frame with the swing parts we are analyzing
        """
        #consider only time between backswing and finish
        halfway_back_ind=self.data['right_wrist_x'].idxmin()
        halfway_front_ind=self.data.right_wrist_x[self.data.index>halfway_back_ind].idxmax()
        middle_data=self.data[(self.data.index>halfway_back_ind)&(self.data.index<halfway_front_ind)]

        if middle_data.empty:
            halfway_back_ind=self.data.right_wrist_x[self.data.index<halfway_back_ind].idxmin()
            halfway_front_ind=self.data.right_wrist_x[self.data.index>halfway_back_ind].idxmax()
            middle_data=self.data[(self.data.index>halfway_back_ind)&(self.data.index<halfway_front_ind)]     
        #find moment of ball contact as the lowest wrist point on y
        contact_frame=middle_data['right_wrist_y'].idxmax()

        #isolate only backswing data
        back_data=self.data[(self.data.index<contact_frame.item())]
        #find moment of top of backswing as the highest wrist point on y
        top_backswing_frame=back_data['right_wrist_y'].idxmin()
        #find moment before start of the swing as the lowest wrist point on y before going halfway back
        halfway_back_data=self.data[self.data.index<halfway_back_ind]
        address_frame=halfway_back_data['right_wrist_y'].idxmax()

        self.data=self.data.iloc[[max([address_frame.item()-4,0]), top_backswing_frame.item(), contact_frame.item()]]
        # print(self.data)
        self.data['index']=self.data.index
        self.data=self.data.reset_index(drop=True)

         
    
    def preprocess_data(self):
        self.load_data()
        # This can be also adjusted when getting the data
        self.data=self.data.reset_index()
        self.data.columns = [col.replace(' X', '_x').replace(' Y', '_y').lower() for col in self.data.columns]
        self.data=self.data.drop(['video_timestamp'],axis=1)

        # Reduce the dataset to only include the 3 phases we are looking for
        self.split_swing()
        


class Evaluator:
    """
    Evaluating the correctness of positions and angles.
    Input dataset has three rows, each for a different swing part.
    """
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.swing_part_indices = {'address': 0, 'top': 1, 'contact': 2}  # Indices for each swing part
        self.head_point_address=[self.data_processor.data.nose_y.iloc[self.swing_part_indices['address']],self.data_processor.data.nose_x.iloc[self.swing_part_indices['address']]]
        self.results={}
    def evaluate_all_swing_parts(self):
        """
        Runs the evaluation for all swing parts automatically and returns a structured result.
        """
        for swing_part, index in self.swing_part_indices.items():
            self.results[swing_part] = self.evaluate_correctness(swing_part, index)
        return self.results

    def evaluate_correctness(self, swing_part, swing_part_id):
        """
        Evaluates the correctness based on the provided swing part and its ID.
        """
        data = self.data_processor.data

        if swing_part == 'address':
            return self.evaluate_address(data, swing_part_id)
        elif swing_part == 'top':
            return self.evaluate_top(data, swing_part_id)
        elif swing_part == 'contact':
            return self.evaluate_contact(data, swing_part_id)


    def evaluate_address(self, data, swing_part_id):
        results = {
            "correct_midpoint": int(data['midpoint_x'].iloc[swing_part_id] - data['left_wrist_x'].iloc[swing_part_id] < 0),
            "correct_arm_angle": int(165 <= data['arm_angle'].iloc[swing_part_id] <= 180)
        }
        return results

    def evaluate_top(self, data, swing_part_id):
        head_point_current = np.array([data['nose_y'].iloc[swing_part_id], data['nose_x'].iloc[swing_part_id]])
        results = {
            "correct_pelvis": int(150 <= data['pelvis_angle'].iloc[swing_part_id] <= 180),
            # "correct_arm_angle": int(150 <= data['arm_angle'].iloc[swing_part_id] <= 180),
            "correct_head": int(self.calculate_head_distance(head_point_current, self.head_point_address) <= 30)
        }
        return results

    def evaluate_contact(self, data, swing_part_id):
        head_point_current = np.array([data['nose_y'].iloc[swing_part_id], data['nose_x'].iloc[swing_part_id]])
        results = {
            "correct_shoulder_ankle": int(data['left_ankle_x'].iloc[swing_part_id] - data['left_shoulder_x'].iloc[swing_part_id] >= 0),
            "correct_head": int(self.calculate_head_distance(head_point_current, self.head_point_address) <= 30),
            "correct_knee_angle": int(165 <= data['knee_angle'].loc[swing_part_id] <= 180),
            "correct_arm_angle": int(160 <= data['arm_angle'].iloc[swing_part_id] <= 180)
        }
        return results

    @staticmethod
    def calculate_head_distance(head_point_current, head_point_address):
        return np.linalg.norm(np.array(head_point_current).flatten()- np.array(head_point_address).flatten())
        




class VideoProcessor:
    """
    Handles frame extraction, writing results into frame and exporting frame
    """
    def __init__(self, folder_path,data_processor,evaluator):
        self.folder_path = folder_path
        self.data=data_processor.data
        self.correct=evaluator.evaluate_all_swing_parts()
        self.swing_part_indices = {'address': 0, 'top': 1, 'contact': 2}  # Indices for each swing part
    
    def print_swing_analysis(self):
        messages = {
            'correct_midpoint': 'WRONG: Arms should be positioned more to the left side of the center of feet.',
            'correct_arm_angle': 'WRONG: Left arm should be straight at this point.',
            'correct_pelvis': 'WRONG: Left ankle, left hip and right shoulder angle should form approximately straight line. Try turning more into the backswing.',
            'correct_head':  'WRONG: The head should remain relatively still until contact.',
            'correct_shoulder_ankle': 'WRONG: Left shoulder should not go beyond the front foot at this point.',
            'correct_knee_angle': 'WRONG: Knee should not bend as much.'
        }
        # Iterate over each swing part and construct the analysis message
        analysis = []
        for swing_part, checks in self.correct.items():
            part_analysis = [f"Swing part {swing_part.upper()}: "]
            messages_list = []
            for check, value in checks.items():
                if value == 0:
                    messages_list.append("-> " + messages[check])

            if messages_list:
                analysis.append("\n".join([part_analysis[0]] + messages_list))

        print("\n\n".join(analysis))

    def save_frame(self):

        video_files = [file for file in os.listdir(self.folder_path) if file.endswith('.mp4')]
        video_path=video_files[0]

        for swing_part, index in self.swing_part_indices.items():
            cap = cv2.VideoCapture(self.folder_path+'/'+video_path)

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.data['index'].iloc[index])
            ret, frame = cap.read()

            frame=self.plot_points(frame, swing_part)

            frame_path =self.folder_path+'/'+ video_path.split(".")[0]+ f'_frame_{swing_part}.jpg'
            cv2.imwrite(frame_path, frame)
            cap.release()

    def plot_line(self, frame, start_point, end_point, is_correct):
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.line(frame, start_point, end_point, color, 2)

    def plot_circle(self, frame, center, is_correct, radius=6, thickness=-1):
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.circle(frame, center, radius, color, thickness)

    def plot_text(self, frame, text, position, is_correct):
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def plot_points(self, frame, swing_part):
        swing_part_id = self.swing_part_indices[swing_part]
        evaluation_results = self.correct[swing_part]
        if swing_part == 'address':
            self.plot_circle(frame, (int(self.data['midpoint_x'].iloc[swing_part_id]), int(self.data['midpoint_y'].iloc[swing_part_id])), evaluation_results['correct_midpoint'])
            self.plot_line(frame, (int(self.data['right_ankle_x'].iloc[swing_part_id]), int(self.data['right_ankle_y'].iloc[swing_part_id])), (int(self.data['left_ankle_x'].iloc[swing_part_id]), int(self.data['left_ankle_y'].iloc[swing_part_id])), evaluation_results['correct_midpoint'])
            self.plot_line(frame, (int(self.data['midpoint_x'].iloc[swing_part_id]), int(self.data['midpoint_y'].iloc[swing_part_id])), (int(self.data['midpoint_x'].iloc[swing_part_id]), int(self.data['midpoint_y'].iloc[swing_part_id]) - 150), evaluation_results['correct_midpoint'])

            arm_angle_text = f'Arm Angle: {self.data["arm_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, arm_angle_text, (10, 120), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), (self.data['left_wrist_x'].iloc[swing_part_id], self.data['left_wrist_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
        elif swing_part == 'contact':
            shoulders_inclination_text = f'Shoulders inclination: {self.data["shoulders_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, shoulders_inclination_text, (10, 20), True)  # True for always yellow
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), True)


            hips_inclination_text = f'Hips inclination: {self.data["hips_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, hips_inclination_text, (10, 45), True)  # True for always yellow
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), True)


            knee_angle_text = f'Knee Angle: {self.data["knee_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, knee_angle_text, (10, 70), evaluation_results['correct_knee_angle'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_knee_x'].iloc[swing_part_id], self.data['left_knee_y'].iloc[swing_part_id]), evaluation_results['correct_knee_angle'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), evaluation_results['correct_knee_angle'])

            self.plot_circle(frame, (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), evaluation_results['correct_shoulder_ankle'])
            self.plot_line(frame, (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id] - 200), evaluation_results['correct_shoulder_ankle'])
            
            if self.data['nose_x'].iloc[swing_part_id]!=0 and self.data['nose_y'].iloc[swing_part_id]!=0:
                self.plot_line(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)
            else:
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)

            arm_angle_text = f'Arm Angle: {self.data["arm_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, arm_angle_text, (10, 120), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), (self.data['left_wrist_x'].iloc[swing_part_id], self.data['left_wrist_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
        
        elif swing_part == 'top':

            shoulders_inclination_text = f'Shoulders inclination: {self.data["shoulders_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, shoulders_inclination_text, (10, 20), True) 
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), True)

            hips_inclination_text = f'Hips inclination: {self.data["hips_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, hips_inclination_text, (10, 45), True)  
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), True)


            pelvis_angle_text = f'Pelvis Angle: {self.data["pelvis_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, pelvis_angle_text, (10, 95), evaluation_results['correct_pelvis'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), evaluation_results['correct_pelvis'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['right_shoulder_x'].iloc[swing_part_id], self.data['right_shoulder_y'].iloc[swing_part_id]), evaluation_results['correct_pelvis'])


            if self.data['nose_x'].iloc[swing_part_id]!=0 and self.data['nose_y'].iloc[swing_part_id]!=0:
                self.plot_line(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)
            else:
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)
            # arm_angle_text = f'Arm Angle: {self.data["arm_angle"].iloc[swing_part_id]:.2f}'
            # self.plot_text(frame, arm_angle_text, (10, 120), evaluation_results['correct_arm_angle'])
            # self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
            # self.plot_line(frame, (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), (self.data['left_wrist_x'].iloc[swing_part_id], self.data['left_wrist_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])

        return frame
   

        
    