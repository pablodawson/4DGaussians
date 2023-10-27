import os
import cv2

data_path = "data/dynerf/cut_roasted_beef/"

# Create a VideoCapture object and read from input file
cams = []

# cam00 cam01 ... cam 20

for i in range(21):
    name = data_path + "cam" + str(i).zfill(2) 
    cam = cv2.VideoCapture(name + ".mp4")

    try:
        os.mkdir(name)
        outpath = os.path.join(name, "images/")
        os.mkdir(outpath)
    except:
        continue

    if (cam.isOpened()== False):
        print("Error opening video stream or file")
    
    # Read until video is completed
    while True:
        ret, frame = cam.read()

        if ret == True:
            frame = cv2.resize(frame, (1352, 1014), interpolation=cv2.INTER_AREA)
            cv2.imwrite(outpath + "/" + str(int(cam.get(cv2.CAP_PROP_POS_FRAMES))).zfill(4) + ".png", frame)
        else:
            break
    
    print(f"Done with {name}")
