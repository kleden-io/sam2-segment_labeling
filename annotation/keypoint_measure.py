import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
import cv2
import rerun as rr

class EstimateKeypoints:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path="models/pose_landmarker.task")
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options, output_segmentation_masks=False)
 
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
        self.gt_aruco_cm = 15

    def input_image(self, image):
        self.image = image
        self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        h, w, _ = image.shape
        self.h = h
        self.w = w

    def detect_aruco(self):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
               self.image , self.aruco_dictionary, parameters=self.aruco_parameters
            )
        
        center = np.mean(corners[0], axis=0)
        size_w = np.max(corners[0][:,0]) - np.min(corners[0][:,0])
        size_h = np.max(corners[0][:,1]) - np.min(corners[0][:,1])

        if corners:
            return corners[0],  center, size_w, size_h
        else: 
            return None

    def get_coordinates_by_index(self, np_results, index):
        for row in np_results:
            if row[0] == index:
                return row[1]
        return None
    
    def track2Dpose(self, results, img_width ,img_height):
        if results.pose_landmarks is None or len(results.pose_landmarks)==0:
            return None
        else: 
            pose_landmarks =  results.pose_landmarks[0]
            normalized_landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
            return np.array([(img_width * lm.x, img_height * lm.y) for lm in normalized_landmarks])


    def get_keypoints(self):
        self.results = self.detector.detect(self.mp_image)
        landmark_2d = self.track2Dpose(results=self.results, img_width=self.w, img_height=self.h)
        return landmark_2d
    
    def get_measurerements(self, results, img_width ,img_height):
        if results.pose_landmarks is None or len(results.pose_landmarks)==0:
            return None
        else: 
            pose_landmarks =  results.pose_landmarks[0]
            normalized_landmarks = [(idx, pose_landmarks[lm]) for idx, lm in enumerate(mp_pose.PoseLandmark)]  
            keypoints = [(idx, (img_width * lm.x, img_height * lm.y)) for idx, lm in normalized_landmarks]
            keypoints_array = np.array(keypoints, dtype=object)

            left_hip = np.array(self.get_coordinates_by_index(keypoints_array, 23))
            right_hip = np.array(self.get_coordinates_by_index(keypoints_array, 24))
            hip_distance = np.linalg.norm(right_hip- left_hip)
            
            left_knee = np.array(self.get_coordinates_by_index(keypoints_array, 25))
            right_knee = np.array(self.get_coordinates_by_index(keypoints_array, 26))
            
            left_ankle = np.array(self.get_coordinates_by_index(keypoints_array, 27))
            right_ankle = np.array(self.get_coordinates_by_index(keypoints_array, 28))
            
            left_leg_distance = np.linalg.norm(left_ankle-left_hip) + np.linalg.norm(left_ankle-left_knee)
            right_leg_distance = np.linalg.norm(right_ankle-right_hip) + np.linalg.norm(right_ankle-right_knee)

            right_shoulder = np.array(self.get_coordinates_by_index(keypoints_array, 12)) 
            left_shoulder = np.array(self.get_coordinates_by_index(keypoints_array, 11))
            left_elbow = np.array(self.get_coordinates_by_index(keypoints_array, 13))
            left_wrist = np.array(self.get_coordinates_by_index(keypoints_array, 15))
            center_chest = (right_shoulder + left_shoulder) / 2
            sleeve_distance = (
                np.linalg.norm(center_chest - left_shoulder) +
                np.linalg.norm(left_shoulder - left_elbow) + 
                np.linalg.norm(left_elbow - left_wrist))
            
            chest_distance = (
                np.linalg.norm( right_shoulder - left_shoulder )
            )

            return hip_distance, left_leg_distance, right_leg_distance, sleeve_distance, chest_distance
        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image
        
    def estimate_pant_size(self, hip_measure):
        """
        based on this site: https://terendesigns.com/pages/measurement-guide
        hip_measure -> in cm
        we multiply by 5 to get the entire hip measure
        """
        total_measure = (hip_measure * 5)/2.54  #  convert to inches

        if (total_measure < 32):
            return "less than 30"
        elif (32<=int(total_measure)<=33):
            return "30"
        elif (34<=int(total_measure<=35)):
            return "32"
        elif (36<=int(total_measure)<=37):
            return "34"
        elif (38<=int(total_measure)<=39):
            return "36"
        elif (40<=int(total_measure)<=41):
            return "38"
        elif (42<=int(total_measure)<=44):
            return "40"
        else:
            return None

    def estimate_top_size_by_sleeve(self, sleeve_measure):
        total_measure = sleeve_measure / 2.54

        if (32<= total_measure <= 33):
            return "small"
        elif(33< total_measure <= 34):
            return "medium"
        elif(34< total_measure <= 35):
            return "large"
        elif(35< total_measure <= 36):
            return "x-large"
        elif(36< total_measure <= 37):
            return "xx-large"
        else:
            return None

    def estimate_top_size_by_chest(self, chest_measure):

        total_measure = chest_measure * 3 / 2.54
        if (total_measure<37):
            return "small"
        elif (37<= total_measure <= 38):
            return "small"
        elif(39< total_measure <= 41):
            return "medium"
        elif(42< total_measure <= 44):
            return "large"
        elif(45< total_measure <= 47):
            return "x-large"
        elif(45< total_measure <= 50):
            return "xx-large"
        else:
            return None
        
    def get_recommendation(self):

        (
            hip_measure,
            left_leg_measure,
            right_leg_measure,
            sleeve_measurement,
            chest_measurement
        ) = self.get_measurerements(results=self.results, img_width=self.w, img_height=self.h)
        
        corners , _, _ , _= self.detect_aruco()
        aruco_dist = np.linalg.norm(corners[0][0] - corners[0][1])

        estimate_hip = hip_measure * 15 / aruco_dist

        estimate_leg_left = left_leg_measure * 15 / aruco_dist
        estimate_leg_right = right_leg_measure * 15 / aruco_dist

        recommeded_pant = self.estimate_pant_size(estimate_hip)

        cm_sleeve = sleeve_measurement * 15 / aruco_dist 
        recommeded_top = self.estimate_top_size_by_sleeve(cm_sleeve)

        estimate_chest = chest_measurement * 15 / aruco_dist
        recommeded_top_chest = self.estimate_top_size_by_chest(estimate_chest)

        RECOMMENDATION = f"""
            # Recommendations
            - Top size: {recommeded_top_chest}
            - Pant: {recommeded_pant}

            # Some calculations from image

            - Estimated chest size: {round(estimate_chest, 2)} cm
            - Estimated left leg size: {round(estimate_leg_right, 2)} cm 
            - Estimated hip size: {round(estimate_hip, 2)} cm
            - Estimated sleeve size: {round(cm_sleeve, 2)} cm
            """
        return RECOMMENDATION

 
"""
rr.init("camera_visualization_demo")
rr.connect()

estimator = EstimateKeypoints()
img = cv2.imread("2001.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
estimator.input_image(rgb)
lm_kp = estimator.get_keypoints()

recommendation = estimator.get_recommendation()

rr.log(
    "video",
    rr.Image(rgb))

rr.log("video/pose/points",
    rr.Points2D(
        lm_kp, 
        keypoint_ids=mp_pose.PoseLandmark))

rr.log("description", 
       rr.TextDocument(str(recommendation).strip(), media_type=rr.MediaType.MARKDOWN), static=True)

"""