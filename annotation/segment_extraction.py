from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sahi.utils.yolov8 import download_yolov8s_model
from ultralytics import YOLO
import torch
import numpy as np
import os 
import ffmpeg

class DetectAndSegment:
    def __init__(self, yolo_model_path, sam2_checkpoint, sam2_model):

        self.yolo_model_path = yolo_model_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model = sam2_model

        torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if os.path.exists(yolo_model_path and sam2_checkpoint and sam2_model):
            self.yolo_model_path = yolo_model_path

            #YOLO init 
            self.model = YOLO(self.yolo_model_path)
            self.model.classes = [0] #only detecting people

            #sam2 init
            self.sam2_checkpoint= sam2_checkpoint
            self.sam2_model = sam2_model
            #video predictor
            self.vid_predictor = build_sam2_video_predictor(self.sam2_model, self.sam2_checkpoint)
            #image predictor        
            self.img_predictor = SAM2ImagePredictor(build_sam2(self.sam2_model, self.sam2_checkpoint))
        else: 
            print("Directory does not exist")

    def download_yolo_model(self):
        
        if not os.path.exists('models'):
            os.makedirs('models')

        download_yolov8s_model(self.yolo_model_path)

    def yolo_inference(self, img_path):
        results = self.model(img_path, stream = False)
        for result in results:
            boxes = result.boxes
        bbox =  boxes.xyxy.tolist()
        return bbox
    
    def video_preprocess(self, video_path):
        output_pattern = 'images/%05d.jpg'
        ffmpeg.input(video_path).output(output_pattern, q=2, start_number=0).run()

    def sam2_video_inference(self, video_path):
        inference_state = self.vid_predictor.init_state(video_path = video_path)
        self.vid_predictor.reset_state(inference_state)

    def sam2_img_inference(self, img):

        bboxes = np.array(self.yolo_inference(img))   
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.img_predictor.set_image(img)
            masks, _, _ = self.img_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes[None, :],
                multimask_output=False,
            )

        return masks
