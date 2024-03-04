import os
import cv2

dirname = "D:/capturas/bici"

videos = os.listdir(dirname)
print(f"Found {len(videos)} videos")
max_frames = 200
start_frame = 2285
prefix = "cam"
camnum = 0
colmap_folder = os.path.join(dirname, "colmap")
os.makedirs(colmap_folder, exist_ok=True)
input_folder = os.path.join(colmap_folder, "images")
os.makedirs(input_folder, exist_ok=True)
ref_frame = 100

for video in videos:

    #if not video.endswith(".mp4") or not video.endswith(".mov"):
    #    continue
    
    name = video[:-4]
    print(f"Processing {name}")

    render_path = os.path.join(dirname, f"{prefix}{camnum:02d}")
    cam = cv2.VideoCapture(os.path.join(dirname, video))
    camnum += 1

    os.makedirs(render_path, exist_ok=True)
    outpath = os.path.join(render_path, "images/")
    os.makedirs(outpath, exist_ok=True)

    # Remove previous images
    for file in os.listdir(outpath):
        os.remove(os.path.join(outpath, file))

    if (cam.isOpened()== False):
        print("Error opening video stream or file")
        continue
    
    # Read until video is completed
    j = 0
    count = 0

    # Skip first frames
    while j < start_frame:
        ret, frame = cam.read()
        j += 1
    
    j= 0

    while count < max_frames:
        ret, frame = cam.read()
        count += 1

        if ret == True:
            h, w, _ = frame.shape
            #frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(outpath + "/" + str(j).zfill(4) + ".png", frame)
            j += 1

            if j == ref_frame:
                cv2.imwrite(os.path.join(input_folder, f"{name}.png"), frame)
        else:
            break

    print(f"Done with {name}")
