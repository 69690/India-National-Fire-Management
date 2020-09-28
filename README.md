# India-National-Safety-Management
Final Year Project for Bennett University
## CCTV
#### Steps:
1. Install all the dependencies
    ```
    pip install -r requirements.txt
    ```
2. Download the model from the link: https://drive.google.com/drive/folders/11miO3Dgo826hWEVEXoykNKFce0UD0K8U?usp=sharing (~240 MB)
    - Place it in the CCTV folder
3. Go to terminal and run the python script yolo.py in CCTV folder
    - Change to CCTV folder
    - Run the following command to run the tutorial
        ```
        python yolo.py --play_video True --video_path videos/fire1.mp4
        ```
4. Command usage
    ```
    usage: yolo.py [-h] [--webcam WEBCAM] [--play_video PLAY_VIDEO]
               [--image IMAGE] [--video_path VIDEO_PATH]
               [--image_path IMAGE_PATH] [--verbose VERBOSE]

    optional arguments:
      -h, --help            show this help message and exit
      --webcam WEBCAM       True/False
      --play_video PLAY_VIDEO
                            Tue/False
      --image IMAGE         Tue/False
      --video_path VIDEO_PATH
                            Path of video file
      --image_path IMAGE_PATH
                            Path of image to detect objects
      --verbose VERBOSE     To print statements
    ```
