import cv2
import numpy as np 
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--show_demo', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--videopath', help="Path of video file", default="demo-video/demo.mp4")
parser.add_argument('--imagepath', help="Path of image to detect objects")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()


def imShow_s(path):
  import cv2
  import matplotlib.pyplot as plt

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

def setup_s():
    fi = open("/home/pi/Desktop/major/config.json", "w+")
    print("Entering configuration menu:")
    Email=str(input("Enter Email name :"))
    pwd=str(input("Enter Password :"))
    fi.writelines(Email)     
    fi.close()
    try:
        user=firebase_admin.auth.create_user(email=Email,password=pwd)
    except:
        print("Failed create an user, Enter valid details or user exists")
        checkSetup()
        return 

    global UID
    UID=user.uid
    ref = db.reference('devices/'+UID)  #+'/value')#.value()
    ref.child("connection").set('0')
    ref.child("fire").set("0")
    ref.child("lastTS").set(datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
    #ref.child("deviceNumber").set('')
    ref.child("fireTS").set('')
    return

def checkSetup_s():
    
    try:
        fi = open("/home/pi/Desktop/major/config.json", "r")
        Email=fi.readline()
        print("User Exisits : "+Email)
        fi.close()
        print("would like enter setup (y/N)")
        
        x=str('N')
        if (x=='y' or x=='Y'):
            setup()
                
    except:
        print("NO json file found")
        setup()

    return 
      

def alertfun_s(msg):#alert on camera fail or disconnection
    li = ["m.phanisai007@gmail.com",
          "santhoshyadav09@gmail.com",
          "help.eceprojects@gmail.com"]
    #li is list of mail Id that gets alerts
    for i in range(len(li)): # iterate through all mail Ids
        s = smtplib.SMTP('smtp.gmail.com', 587) 
        s.starttls() 
        s.login("help.eceprojects@gmail.com", "vidyajyothi@03")
        #logins to a mailId (admin) to send mail
        
        sub="CCTV Alert !" #subject
        message = 'Subject: {}\n\n{}'.format(sub, msg)
        print(message)
        s.sendmail("help.eceprojects@gmail.com", li[i], message)
        s.quit()

def camConnCheck_s():
    print("camconncheck")
    global connflag
    if(connflag):
        flag="0"
    else:
        flag="1"
    ref = db.reference('devices/'+UID)  #+'/value')#.value()
    ref.child("connection").set(flag)
    time.sleep(10)
    camConnCheck()


def UploadImg_s():
    time.sleep(10)
    global UID
    bucket = firebase_admin.storage.bucket("cv-cam.appspot.com")
    imageBlob = bucket.blob(UID+'/profile.jpg')
    res=imageBlob.upload_from_filename(r'/home/pi/Desktop/major/profile.jpg')
    #upload_blob('profile.JPG','profile.jpg')
    print("uploaded")
    ref = db.reference('devices/'+UID)  #+'/value')#.value()
    ref.child("ImgTS").set(str(datetime.now()))
    time.sleep(110)
    UploadImg()

def uploadfireImg_s():
    global UID
    bucket = firebase_admin.storage.bucket("cv-cam.appspot.com")
    imageBlob = bucket.blob(UID+'/fire.jpg')
    res=imageBlob.upload_from_filename(r'/home/pi/Desktop/major/fire.jpg')
    #upload_blob('profile.JPG','profile.jpg')

    
def fireCheck_s():
    global UID
    global fireflag
    print("firecheck")
    if(fireflag):
        ref = db.reference('devices/'+UID)  #+'/value')#.value()
        ref.child("fire").set("1")
        ref = db.reference('devices/'+UID)  #+'/value')#.value()
        ref.child("fireTS").set(str(datetime.now()))
        uploadfireImg()
    else:
        ref = db.reference('devices/'+UID)  #+'/value')#.value()
        ref.child("fire").set("0")
        ref = db.reference('devices/'+UID)  #+'/value')#.value()
        ref.child("fireTS").set("Not Applicable")
    #algo

    #on fire detect update fireTS and fire
    
    fireCheck()

def LatestTS_s():
    print("lastTS")
    global UID
    ref = db.reference('devices/'+UID)  #+'/value')#.value()
    ref.child("lastTS").set(str(datetime.now()))
    time.sleep(60)
    LatestTS()

def algo_s():
    fire_cascade = cv2.CascadeClassifier('/home/pi/Desktop/major/fire_detection.xml')
    t=0
    t1=0
    t2=0
    cap = cv2.VideoCapture(0)
    while 1:
        if(cap.isOpened()==False):
            global connflag
            connflag=1
            if((datetime.now().minute-t2)>30):
                    t2=datetime.now().minute
                    msg = "FIRE DETECTED by CV - CCTV Camera !\n\n\nNOTE: This auto system generated mail on disconnection of computer vision camera";
                    alertfun(msg)# send mail
            #mail as well
        ret, img = cap.read()
        if not ret:
            print("no image")
            continue
        if((datetime.now().minute-t)>1):
            cv2.imwrite('profile.jpg',img)
            print("image saved")
            t=datetime.now().minute

        cv2.imshow('orignal Video',img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        fire = fire_cascade.detectMultiScale(img, 1.2, 5)
        for (x,y,w,h) in fire:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            print ('Fire is detected..!')
            cv2.imwrite('fire.jpg',img)
            time.sleep(0.2)
            global fireflag
            fireflag=1
            if((datetime.now().minute-t1)>30):
                t1=datetime.now().minute
                msg = "FIRE DETECTED by CV Camera !\n\n\nNOTE: This auto system generated mail on disconnection of computer vision camera";
                alertfun(msg)# send mail
        #cv2.imshow('img',img)
        
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def save_images_s(datasets, path):
    part_1_img_numpy = datasets[:35]
    part_2_img_numpy = datasets[35:80]
    part_3_img_numpy = datasets[80:110]
    imgs = [part_1_img_numpy, part_2_img_numpy, part_3_img_numpy]
    names = ['part1','part2','part3']
    
    for name,x in zip(names,imgs):
        np.save(path+'/'+name, x)
        #print(path+'/'+name, x)


# use this to upload files
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)

# use this to download a file  
def download(path):
  from google.colab import files
  files.download(path)

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("obj.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	img=cv2.resize(img, (800,600))
	cv2.imshow("Image", img)

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)

	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)

		key = cv2.waitKey(1)
		if cv2.waitKey(1) & 0xFF ==ord('q'):
			break
	cap.release()

if __name__ == '__main__':
	webcam = args.webcam
	video_play = args.show_demo
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		videopath = args.videopath
		if args.verbose:
			print('Opening '+videopath+" .... ")
		start_video(videopath)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+imagepath+" .... ")
		image_detect(imagepath)
	

	cv2.destroyAllWindows()