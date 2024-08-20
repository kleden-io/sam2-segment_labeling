# How to use
This project demonstrates the use of a camera visualization tool to estimate keypoints from an image and log the results.
## Usage

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
