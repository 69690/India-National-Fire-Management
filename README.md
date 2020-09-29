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
    python detect.py [optional arguments]
   
    optional arguments:
      -h, --help            
      --webcam WEBCAM       
      --play_video PLAY_VIDEO
      --image IMAGE        
      --video_path VIDEO_PATH
      --image_path IMAGE_PATH
  
    ```
5. Model.ipynb is included in this repository, the other links are as follows:
    - Trained Model Download: https://drive.google.com/drive/folders/11miO3Dgo826hWEVEXoykNKFce0UD0K8U?usp=sharing
    - Dataset Download: https://drive.google.com/file/d/1ytEjCJOHToOP9j1dvem-S7ID9MCmYLWu/view?usp=sharing
