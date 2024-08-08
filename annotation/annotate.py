import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
import yaml



def main():
    with open('config.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    if os.path.exists(config['sam2']['model'] and config['sam2']['config']): 
        sam2_checkpoint = config['sam2']['model']
        model_cfg = config['sam2']['config']
        predictor = build_sam2(model_cfg, sam2_checkpoint)
    else: 
        print('files does not exist')

if __name__ == "__main__":
    main()
