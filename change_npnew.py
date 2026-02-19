import torch
import numpy as np
import cv2
import os
import torchvision
from dwpose import DwposeDetector

import csv

# DWpose 모델 초기화 (최초 실행 시 모델 자동 다운로드)
#1024 576   512 288    256 144    128 72   16 9
model = DwposeDetector.from_pretrained_default()
resize = torchvision.transforms.Resize((144, 256))
def get_keypoints(image_path):
    
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (256, 144))

    # DWpose로 keypoints 추출
    imgOut,j,source = model(image_rgb,
    include_hand=True,
    include_face=True,
    include_body=True,
    image_and_json=True,
    detect_resolution=512)

    error_pass = False
    pose_kpts = None
    hand_left = None
    hand_right =None
    face = None
    try:
        if j['people'][1]:
           pose_kpts = np.zeros((18, 3), dtype=np.float32)
           hand_left = np.zeros((21, 3), dtype=np.float32)
           hand_right = np.zeros((21, 3), dtype=np.float32)
           face = np.zeros((70, 3), dtype=np.float32)
           error_pass = True 
    except:
        try:
            pose_kpts = np.array(j['people'][0]['pose_keypoints_2d'])
        except:
            pass
        if pose_kpts is None:
            pose_kpts = np.zeros((18, 3), dtype=np.float32)
            error_pass = True
        elif pose_kpts.size == 18 * 3:
            pose_kpts = pose_kpts.reshape(18, 3)
        else:
            pose_kpts = np.zeros((18, 3), dtype=np.float32)
            error_pass = True
        
        try:
            hand_left = np.array(j['people'][0]['hand_left_keypoints_2d'], dtype=np.float32)
        except:
            pass
        if hand_left is None:
            hand_left = np.zeros((21, 3), dtype=np.float32)
        elif hand_left.size == 21 * 3:
            hand_left = hand_left.reshape(21, 3)
        else:
            hand_left = np.zeros((21, 3), dtype=np.float32)
        
        try:
            hand_right = np.array(j['people'][0]['hand_right_keypoints_2d'], dtype=np.float32)
        except:
            pass
        if hand_right is None:
            hand_right = np.zeros((21, 3), dtype=np.float32)
        elif hand_right.size == 21 * 3:
            hand_right = hand_right.reshape(21, 3)
        else:
            hand_right = np.zeros((21, 3), dtype=np.float32)
            
        try:
            face = np.array(j['people'][0]['face_keypoints_2d'], dtype=np.float32)
        except:
            pass
        if face is None:
            face = np.zeros((70, 3), dtype=np.float32)
            error_pass = True
        elif face.size == 70 * 3:
            face = face.reshape(70, 3)
        else:
            face = np.zeros((70, 3), dtype=np.float32)
            error_pass = True
    kpts = np.concatenate([pose_kpts, hand_left, hand_right, face], axis=0)  # (130, 3)
    def keypoints_to_heatmaps(keypoints, height, width, h2, w2, sigma=1):
        heatmaps = np.zeros((keypoints.shape[0], height, width), dtype=np.uint8)  #130, 128, 256
        for idx, (x, y, v) in enumerate(keypoints):
            if v < 0.05:
                continue
            x = int(x*width/w2)
            y = int(y*height/h2)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            
            cv2.circle(heatmaps[idx], (x, y), sigma, 1, -1)
        return torch.tensor(heatmaps, dtype=torch.uint8)  # [127, H, W]

    h, w = image_rgb.shape[:2]
    h2 = np.array(imgOut).shape[0]
    w2 = np.array(imgOut).shape[1]

    pose_tensor_heatmap = keypoints_to_heatmaps(kpts, h, w, h2, w2)*255
    out_pose = resize(torch.from_numpy(np.array(imgOut)).permute(2, 0, 1))  # [3, 128, 256]
    image_rgb_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
    return image_rgb_tensor, pose_tensor_heatmap, error_pass, out_pose
origin_image_tensors = torch.tensor([])
pose_tensors = torch.tensor([])
pose_tensors2 = torch.tensor([])
output_image_tensors = torch.tensor([])
out_pose_tensors  = torch.tensor([])
out_pose2_tensors  = torch.tensor([])

save_number = 1



root1 = ""
root2 = ""
i_number = 1
# trainlst_file = open("train.lst", mode='r')
# trainlsts=trainlst_file.readlines()

with open('fasion-resize-pairs-train.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader) # 첫 행 건너뛰기 
    for row in csv_reader: # 두번째 행부터 행별로 읽기
        root1_ok = True
        root2_ok = True
        
        # for trainlst in trainlsts:
        #     if(trainlst.strip() == row[0]):
        #         root1_ok = True
                
        # for trainlst in trainlsts:
        #     if(trainlst.strip() == row[1]):
        #         root2_ok = True
                            
        if 'back' in row[0]:
            root1_ok = False
        if 'back' in row[1]:
            root2_ok = False
        if 'flat' in row[0]:
            root1_ok = False
        if 'flat' in row[1]:
            root2_ok = False
        if 'additional' in row[0]:
            root1_ok = False
        if 'additional' in row[1]:
            root2_ok = False

        
        if root1_ok == True and root2_ok == True:
            root1 = ""
            root2 = ""
            root1 = row[0][0:7] + "/"
            if row[0][7:12] == "WOMEN":
                root1 = root1 + "WOMEN" + "/"
                row_split = row[0][12:].split('id0')
                root1 = root1 + row_split[0] + "/" + "id_0" + row_split[1][0:7] + "/" + row_split[1][7:11] + "_" + row_split[1][11:]
            
            elif row[0][7:10] == "MEN":
                root1 = root1 + "MEN" + "/"
                row_split = row[0][10:].split('id0')
                root1 = root1 + row_split[0] + "/" + "id_0" + row_split[1][0:7] + "/" + row_split[1][7:11]+ "_" + row_split[1][11:]
            
            root2 = row[1][0:7] + "/"
            if row[1][7:12] == "WOMEN":
                root2 = root2 + "WOMEN" + "/"
                row_split = row[1][12:].split('id0')
                root2 = root2 + row_split[0] + "/" + "id_0" + row_split[1][0:7] + "/" + row_split[1][7:11]+ "_" + row_split[1][11:]
            
            elif row[1][7:10] == "MEN":
                root2 = root2 + "MEN" + "/"
                row_split = row[1][10:].split('id0')
                root2 = root2 + row_split[0] + "/" + "id_0" + row_split[1][0:7] + "/" + row_split[1][7:11]+ "_" + row_split[1][11:]
            

        
            
            image_path = f"./" + root2
            image_path2 = f"./" + root1
            
            image_bgr = cv2.imread(image_path2)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (256, 144))
            image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
            output_image_tensor, pose_tensor, error_pass, out_pose = get_keypoints(image_path)
            # print(pose_tensor.shape)
            output_image_tensor2, pose_tensor2, error_pass2, out_pose2 = get_keypoints(image_path2)
            # pose_img = (pose_tensor).max(dim=0)[0].cpu().numpy()  # [64, 128], 0~1
            pose_tensor = (pose_tensor).type(torch.uint8)
            pose_tensor2 = (pose_tensor2).type(torch.uint8)
            if pose_tensor.shape == torch.Size([130, 144, 256]) and pose_tensor2.shape == torch.Size([130, 144, 256]) and output_image_tensor.shape == torch.Size([3, 144, 256]) and error_pass==False and error_pass2==False:
                print(f'good {i_number}')

                pose_tensor = pose_tensor.unsqueeze(0)
                pose_tensor2 = pose_tensor2.unsqueeze(0)
                output_image_tensor = output_image_tensor.unsqueeze(0)
                out_pose_tensor = out_pose.unsqueeze(0)
                out_pose2_tensor = out_pose2.unsqueeze(0)

                origin_image_tensors = torch.cat((origin_image_tensors, output_image_tensor2.unsqueeze(0)), dim=0) 
                pose_tensors = torch.cat((pose_tensors, pose_tensor), dim=0) 
                pose_tensors2 = torch.cat((pose_tensors2, pose_tensor2), dim=0) 
                out_pose_tensors = torch.cat((out_pose_tensors, out_pose_tensor), dim=0) 
                out_pose2_tensors = torch.cat((out_pose2_tensors, out_pose2_tensor), dim=0) 
                output_image_tensors = torch.cat((output_image_tensors, output_image_tensor), dim=0) 

            else:
                print(f'bad {i_number}')
            i_number += 1
            if origin_image_tensors.shape[0] >= 20:
                np.save(f'./outdate/origin_img_{save_number}.npy', origin_image_tensors.numpy().astype(np.uint8))
                np.save(f'./outdate/output_img_{save_number}.npy', output_image_tensors.numpy().astype(np.uint8))
                np.save(f'./outdate/pose_{save_number}.npy', pose_tensors.numpy().astype(np.uint8))
                np.save(f'./outdate/pose2_{save_number}.npy', pose_tensors2.numpy().astype(np.uint8))
                np.save(f'./outdate/pose_img_{save_number}.npy', out_pose_tensors.numpy().astype(np.uint8))
                np.save(f'./outdate/pose2_img_{save_number}.npy', out_pose2_tensors.numpy().astype(np.uint8))
                origin_image_tensors = torch.tensor([])
                pose_tensors = torch.tensor([])
                pose_tensors2 = torch.tensor([])
                output_image_tensors = torch.tensor([])
                out_pose_tensors  = torch.tensor([])
                out_pose2_tensors  = torch.tensor([])
                save_number +=1
   
np.save(f'./outdate/origin_img_{save_number}.npy', origin_image_tensors.numpy().astype(np.uint8))
np.save(f'./outdate/output_img_{save_number}.npy', output_image_tensors.numpy().astype(np.uint8))
np.save(f'./outdate/pose_{save_number}.npy', pose_tensors.numpy().astype(np.uint8))
np.save(f'./outdate/pose2_{save_number}.npy', pose_tensors2.numpy().astype(np.uint8))
np.save(f'./outdate/pose_img_{save_number}.npy', out_pose_tensors.numpy().astype(np.uint8))
np.save(f'./outdate/pose2_img_{save_number}.npy', out_pose2_tensors.numpy().astype(np.uint8))
origin_image_tensors = torch.tensor([])
pose_tensors = torch.tensor([])
pose_tensors2 = torch.tensor([])
output_image_tensors = torch.tensor([])
out_pose_tensors  = torch.tensor([])
out_pose2_tensors  = torch.tensor([])
        