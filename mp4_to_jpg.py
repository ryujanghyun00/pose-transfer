import torch
import numpy as np
import cv2
import os

save_number_dir = 0
for i in range(10):
    cap = cv2.VideoCapture(f"./data3_2/{i+1}.mp4")
    save_number = 0
    
    if not os.path.exists(f'./data3_2/{save_number_dir:04d}'):
        os.makedirs(f'./data3_2/{save_number_dir:04d}')
    frame_count = 1
    while True:
        ret, frame = cap.read()
        frame_count +=1
        if not ret:
            break  # 동영상 끝에 도달하면 종료

        # 20 프레임 단위로 저장
        if frame_count % 15 == 0:
            frame_filename = os.path.join(f'./data3_2/{save_number_dir:04d}/{save_number:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"저장: {frame_filename}")
            save_number +=1
    save_number_dir +=1