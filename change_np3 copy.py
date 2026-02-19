import torch
import numpy as np
import cv2
import os
from dwpose import DwposeDetector
import torchvision
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

i_number = None

###########
save_number = 3336
for i in range(10):
    i_number = i+1
    image_number = 2
    
    while True:
        if os.path.isfile(f"./data/{i_number}/output_{image_number:05d}_.png"):
            pass
        else:
            break
        image_path = f"./data/{i_number}/output_{image_number:05d}_.png"
        image_path2 = f"./data/{i_number}/original_00001_.png"
        output_image_tensor, pose_tensor, error_pass, out_pose = get_keypoints(image_path)
        output_image_tensor2, pose_tensor2, error_pass2, out_pose2 = get_keypoints(image_path2)
        pose_tensor = (pose_tensor).type(torch.uint8)
        pose_tensor2 = (pose_tensor2).type(torch.uint8)
        if pose_tensor.shape == torch.Size([130, 144, 256]) and pose_tensor2.shape == torch.Size([130, 144, 256]) and output_image_tensor.shape == torch.Size([3, 144, 256]) and error_pass==False and error_pass2==False:
            print(f'good {i_number}, {image_number}, {error_pass}')     
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
            print('bad')        
        image_number += 1
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
        