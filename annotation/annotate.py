from segment_extraction import DetectAndSegment
from keypoint_measure import EstimateKeypoints
import mediapipe.python.solutions.pose as mp_pose
import yaml
import os
import tqdm
import cv2
import rerun
import numpy as np

class GenerateDataset:
    def __init__(self, 
                    recording_path, 
                    output_path,
                    yolo_path,
                    sam2_ckp,
                    sam2_cfg
                    ):
        self.recording_path = recording_path
        self.output_path = output_path

        self.segmentor = DetectAndSegment(yolo_model_path=yolo_path, 
            sam2_checkpoint=sam2_ckp, 
            sam2_model=sam2_cfg
            )
        
        self.tracker_landmark = EstimateKeypoints()


    def process_directories(self):

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        file_list = os.listdir(self.recording_path)
        for sample in file_list:
            images_list =  os.listdir(os.path.join(self.recording_path, sample, 'images'))
            video_list = os.listdir(os.path.join(self.recording_path, sample, 'video'))
            print(f"Got for {sample}")
            for image in images_list:
                img_path = os.path.join(self.recording_path, sample, 'images', image)
                # self.process_images(img_path)
                print(img_path)

            for video in video_list:
                video_path = os.path.join(self.recording_path, sample, 'video', video)
                self.process_video(sample=video_path, iteration=1)

    def process_images(self, sample, iteration):
        image = cv2.imread(sample)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.segmentor.sam2_img_inference(image)
        rerun.log("image", rerun.Image(np.array(image))) 
        # rerun.log("mask", rerun.Tensor(mask[0]*255))
        rerun.log("image/mask", rerun.SegmentationImage(mask[0]*255))

    def process_video(self, sample, iteration):
        cap = cv2.VideoCapture(sample)
        recommended = False
        print(f"Working with path {sample}")
        for i,_ in enumerate(iter(lambda: cap.read(), (False, None))):
            ret,frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.tracker_landmark.input_image(frame)

            landmark_pose = self.tracker_landmark.get_keypoints()
            if i < 20 or not recommended:
                try:
                    recommendation = self.tracker_landmark.get_recommendation()
                    recommended = True
                    print("Recomendation done")
                except Exception as e:
                    print(e)

            mask = self.segmentor.sam2_img_inference(frame)
            frame = self.tracker_landmark.annonimaze_face(frame)
            rerun.log("image", rerun.Image(np.array(frame)))
            rerun.log("image/mask", rerun.SegmentationImage(mask[0]*255)) 
            rerun.log("image/pose/points",
                    rerun.Points2D(
                        landmark_pose, 
                        keypoint_ids=mp_pose.PoseLandmark))
            if recommended:
                rerun.log("description", 
                            rerun.TextDocument(str(recommendation).strip(), 
                                               media_type=rerun.MediaType.MARKDOWN), 
                                               static=True)
            

        
def main():
    with open('config.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    if os.path.exists(config['sam2']['model'] and config['sam2']['config']): 
        sam2_checkpoint = config['sam2']['model']
        model_cfg = config['sam2']['config']
        yolo_model = config['yolo']['modelpath']
        recording_path = config['recording']['path']
        output_path = config['save_dataset']['path']

        print(sam2_checkpoint)
        print(model_cfg)
        print(yolo_model)

        rerun.init("camera_visualization_demo")
        rerun.connect()
        
        gen_dataset = GenerateDataset(
            recording_path=recording_path, 
            output_path=output_path,
            yolo_path=yolo_model,
            sam2_ckp=sam2_checkpoint,
            sam2_cfg=model_cfg
            )
        
        gen_dataset.process_directories()

        rerun.disconnect()


    else: 
        print('files does not exist')

if __name__ == "__main__":
    main()
