import os
import cv2
from concurrent.futures import ProcessPoolExecutor

def extract_video(vid_path, out_dir):
    vid_name = os.path.basename(vid_path).split('.')[0]
    save_images_path = os.path.join(out_dir, vid_name)
    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path)
    
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    frame_num = 0
    img_count = 0
    while success:
        img_path = os.path.join(save_images_path, "{:05d}.png".format(frame_num))
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, image)
            img_count += 1
        success, image = vidcap.read()
        frame_num += 1
    
    return f"{vid_name} done: {img_count} frames extracted."

def main():
    clips_dir = "JAAD/JAAD_clips"
    out_dir = "JAAD/images"
    
    videos = [os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.endswith('.mp4')]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result in executor.map(extract_video, videos, [out_dir]*len(videos)):
            print(result)

if __name__ == '__main__':
    main()
