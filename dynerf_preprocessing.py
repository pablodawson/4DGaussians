import os
import cv2

data_path = "data/dynerf/sear_steak/"

# Create a VideoCapture object and read from input file
cams = []

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
    j = 0
    while True:
        ret, frame = cam.read()

        if ret == True:
            frame = cv2.resize(frame, (1352, 1014), interpolation=cv2.INTER_AREA)
            cv2.imwrite(outpath + "/" + str(j).zfill(4) + ".png", frame)
            j += 1
        else:
            break
    
    print(f"Done with {name}")
