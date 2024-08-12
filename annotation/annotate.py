from segment_extraction import DetectAndSegment
from sklearn.model_selection import train_test_split
import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GenerateDataset:
    def __init__(self, 
                recording_path, 
                output_path, 
                test_size = 0.2, 
                val_size=0.1, 
                random_state = 42):
        self.recording_path = recording_path
        self.output_path = output_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def process_samples(self):

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        file_list = os.listdir(self.recording_path)
        


def main():
    with open('config.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    if os.path.exists(config['sam2']['model'] and config['sam2']['config']): 
        sam2_checkpoint = config['sam2']['model']
        model_cfg = config['sam2']['config']
        yolo_model = config['yolo']['modelpath']

        print(sam2_checkpoint)
        print(model_cfg)
        print(yolo_model)

        #init

        segmentor = DetectAndSegment(
            yolo_model_path=yolo_model, 
            sam2_checkpoint=sam2_checkpoint, 
            sam2_model=model_cfg
            )
        
        if not os.path.exists(yolo_model):
            segmentor.download_yolo_model()

        # load an image
        image = cv2.imread('me.jpeg')
        image =  cv2.resize(image, (480,720))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = segmentor.sam2_img_inference(rgb_image)
        
        
        cv2.imshow("frame", image)
        cv2.imshow("mask", mask[0]*255)
        cv2.waitKey()


    else: 
        print('files does not exist')

if __name__ == "__main__":
    main()
