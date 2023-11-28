import os
import cv2
import argparse

def main(data_path):
    cams = []

    for i in range(46):
        if i == 0:
            continue
        name = data_path + "/camera_" + str(i).zfill(4)
        print(name)
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
                h, w, _ = frame.shape
                frame = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
                cv2.imwrite(outpath + "/" + str(j).zfill(4) + ".png", frame)
                j += 1
            else:
                break
        
        print(f"Done with {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, default="/media/pablo/Nuevo vol/google_im/welder/")
    args = parser.parse_args()

    main(args.s)