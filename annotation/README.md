# How to use
This project demonstrates the use of a camera visualization tool to estimate keypoints from an image and log the results.
## Usage for keypoint_measure.py

1. Initialize the camera visualization demo:
    ```python
    rr.init("camera_visualization_demo")
    rr.connect()
    ```

2. Create an instance of the keypoint estimator:
    ```python
    estimator = EstimateKeypoints()
    ```

3. Read and process the input image:
    ```python
    img = cv2.imread("2001.jpg")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    estimator.input_image(rgb)
    ```

4. Get the keypoints from the estimator:
    ```python
    lm_kp = estimator.get_keypoints()
    ```

5. Get recommendations based on the keypoints:
    ```python
    recommendation = estimator.get_recommendation()
    ```

6. Log the input image:
    ```python
    rr.log("video", rr.Image(rgb))
    ```

7. Log the keypoints:
    ```python
    rr.log("video/pose/points", rr.Points2D(lm_kp, keypoint_ids=mp_pose.PoseLandmark))
    ```

8. Log the recommendations:
    ```python
    rr.log("description", rr.TextDocument(str(recommendation).strip(), media_type=rr.MediaType.MARKDOWN), static=True)
    ```

## Explanation

- `rr.init("camera_visualization_demo")` and `rr.connect()` initialize and connect to the camera visualization tool.
- `EstimateKeypoints()` creates an instance of the keypoint estimator.
- `cv2.imread("2001.jpg")` reads the input image, and `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` converts it to RGB format.
- `estimator.input_image(rgb)` inputs the RGB image to the estimator.
- `estimator.get_keypoints()` retrieves the keypoints from the image.
- `estimator.get_recommendation()` gets recommendations based on the keypoints.
- `rr.log("video", rr.Image(rgb))` logs the input image.
- `rr.log("video/pose/points", rr.Points2D(lm_kp, keypoint_ids=mp_pose.PoseLandmark))` logs the keypoints.
- `rr.log("description", rr.TextDocument(str(recommendation).strip(), media_type=rr.MediaType.MARKDOWN), static=True)` logs the recommendations as a text document.

# Image Segmentation Demo

This project demonstrates the use of a YOLO model and SAM2 for image segmentation. The code initializes the segmentor, loads an image, performs segmentation, and displays the results.

## Usage for segment_extraction.py

1. **Initialization**:
    ```python
    # Initialize the segmentor with the YOLO and SAM2 model paths
    segmentor = DetectAndSegment(
        yolo_model_path=yolo_model, 
        sam2_checkpoint=sam2_checkpoint, 
        sam2_model=model_cfg
    )
    ```

2. **Download YOLO model if not available**:
    ```python
    if not os.path.exists(yolo_model):
        segmentor.download_yolo_model()
    ```

3. **Load and preprocess the image**:
    ```python
    image = cv2.imread('me.jpeg')
    image = cv2.resize(image, (480, 720))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ```

4. **Perform image segmentation**:
    ```python
    mask = segmentor.sam2_img_inference(rgb_image)
    ```

5. **Display the results**:
    ```python
    cv2.imshow("frame", image)
    cv2.imshow("mask", mask[0] * 255)
    cv2.waitKey()
    ```
## Integration

To run everithing, simply just run: 

```
python annotate.py
```

This will automatically connect to Rerun session (must be already openned). For remote connection, please add the IP address inside:

```
rerun.connect('IP address')
```

## Usage

1. **Initialize the camera visualization demo**:
    ```python
    rerun.init("camera_visualization_demo")
    rerun.connect()
    ```

2. **Create an instance of the dataset generator**:
    ```python
    gen_dataset = GenerateDataset(
        recording_path=recording_path, 
        output_path=output_path,
        yolo_path=yolo_model,
        sam2_ckp=sam2_checkpoint,
        sam2_cfg=model_cfg
    )
    ```

3. **Process directories to generate the dataset**:
    ```python
    gen_dataset.process_directories()
    ```

4. **Disconnect from the camera visualization tool**:
    ```python
    rerun.disconnect()
    ```
