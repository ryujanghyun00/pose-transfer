import cv2
import numpy as np
from dwpose import DwposeDetector
import torch
import torchvision
import tkinter as tk
from PIL import Image, ImageTk
import torch.nn as nn

def get_keypoints(image_rgb):
    # DWpose로 keypoints 추출
    image_rgb = image_rgb
    imgOut,j,source = model(image_rgb,
    include_hand=True,
    include_face=True,
    include_body=True,
    image_and_json=True,
    detect_resolution=512)

    out_pose = resize(torch.from_numpy(np.array(imgOut)).permute(2, 0, 1))  # [3, 128, 256]
    image_rgb_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
    return image_rgb_tensor, out_pose

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)


def downconv2x(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)


def upconv2x(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)


class ResidualBlock(nn.Module):
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        layers = [
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x) + x
        return y

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, (height * width), d_model))
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
    def forward(self, x):
        return x + self.pos_embed
class Generator(nn.Module):
    def __init__(self, in1_channels = 3, in2_channels=6, out_channels=3, ngf=32, nhead = 8, num_layers = 1):
        super().__init__()
        self.in1_conv1 = self.inconv(in1_channels, ngf)
        self.in1_down1 = self.down2x(ngf, ngf*2)
        self.in1_down2 = self.down2x(ngf*2, ngf*4)
        self.in1_down3 = self.down2x(ngf*4, ngf*8)

        self.in2_conv1 = self.inconv(in2_channels, ngf)
        self.in2_down1 = self.down2x(ngf, ngf*2)
        self.in2_down2 = self.down2x(ngf*2, ngf*4)
        self.in2_down3 = self.down2x(ngf*4, ngf*8)
        
        self.out_up2 = self.up2x(ngf*8, ngf*4)
        self.out_up3 = self.up2x(ngf*4, ngf*2)
        self.out_up4 = self.up2x(ngf*2, ngf)
        self.out_conv1 = self.outconv(ngf, out_channels)
            
        self.pos_encoding1_1 = PositionalEncoding2D(ngf*8, 22, 32)
        self.pos_encoding1_2 = PositionalEncoding2D(ngf*8, 22, 32)

        self.transformer1 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )
            

        self.pos_encoding2_1 = PositionalEncoding2D(ngf*8, 22, 32)
        self.pos_encoding2_2 = PositionalEncoding2D(ngf*8, 22, 32)

        self.transformer2 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )


        self.pos_encoding3_1 = PositionalEncoding2D(ngf*8, 22, 32)
        self.pos_encoding3_2 = PositionalEncoding2D(ngf*8, 22, 32)

        self.transformer3 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )


        self.pos_encoding4_1 = PositionalEncoding2D(ngf*8, 22, 32)
        self.pos_encoding4_2 = PositionalEncoding2D(ngf*8, 22, 32)

        self.transformer4 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )
     
    
    def inconv(self, in_channels, out_channels):
        return nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def outconv(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            conv1x1(in_channels, out_channels),
            nn.Tanh()
        )
    
    def down2x(self, in_channels, out_channels):
        return nn.Sequential(
            downconv2x(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def up2x(self, in_channels, out_channels):
        return nn.Sequential(
            upconv2x(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    def forward(self, origin_img, pose_img, origin_pose_img):
        
        pose_img=torch.cat([pose_img, origin_pose_img], dim =1)
        src_feat1 = self.in1_conv1(origin_img)
        src_feat2 = self.in1_down1(src_feat1)
        src_feat3 = self.in1_down2(src_feat2)
        src_feat4 = self.in1_down3(src_feat3)
     
        tgt_feat1 = self.in2_conv1(pose_img)
        tgt_feat2 = self.in2_down1(tgt_feat1)
        tgt_feat3 = self.in2_down2(tgt_feat2)
        tgt_feat4 = self.in2_down3(tgt_feat3)
    
        src_seq = src_feat4.flatten(2).permute(0,2,1)
        tgt_seq = tgt_feat4.flatten(2).permute(0,2,1)
        
        out_seq = self.pos_encoding1_1(src_seq)
        tgt_seq = self.pos_encoding1_2(tgt_seq)
        out_seq1 = self.transformer1(out_seq, tgt_seq)
        out_seq1 = out_seq1 + src_seq
    
        out_seq = self.pos_encoding2_1(out_seq1)
        tgt_seq = self.pos_encoding2_2(tgt_seq)
        out_seq2 = self.transformer2(out_seq, tgt_seq)
        out_seq2 = out_seq2 + out_seq1
        

        out_seq = self.pos_encoding3_1(out_seq2)
        tgt_seq = self.pos_encoding3_2(tgt_seq)
        out_seq3 = self.transformer3(out_seq, tgt_seq)
        out_seq3 = out_seq3 + out_seq2
      

        out_seq = self.pos_encoding4_1(out_seq3)
        tgt_seq = self.pos_encoding4_2(tgt_seq)
        out_seq4 = self.transformer4(out_seq, tgt_seq)
        out_seq4 = out_seq4 + out_seq3
      
        out_feat = out_seq4.permute(0, 2, 1).reshape(src_feat4.shape)
        out_img = self.out_conv1(self.out_up4(self.out_up3(self.out_up2(out_feat))))
        return out_img

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

gen = torch.load('./pth/train8gen120001.pt', weights_only=False)  

model = DwposeDetector.from_pretrained_default()
resize = torchvision.transforms.Resize((256, 176))

root = tk.Tk()
root.title("Generated Images")
img_label1 = tk.Label(root)
img_label2 = tk.Label(root)
img_label3 = tk.Label(root)
img_label4 = tk.Label(root)
img_label5 = tk.Label(root)
img_label6 = tk.Label(root)
img_label1.grid(row=0, column=0)
img_label2.grid(row=0, column=1)
img_label3.grid(row=1, column=0)
img_label4.grid(row=1, column=1)
img_label5.grid(row=2, column=1)
img_label6.grid(row=2, column=0)


    
image_bgr = cv2.imread('00000.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image_rgb, (176, 256))
image_rgb_tensor_origin, out_pose_origin = get_keypoints(image_rgb)

cap = cv2.VideoCapture(0)
#1920 x 1080      x : 1080 = 114 : 256     123120 = 256x  x = 480.93   640 : 480     x:480 = 114 : 256   54720  =256x   x = 213.75
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 176)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (176, 256))

    image_rgb_tensor, out_pose = get_keypoints(frame)
    

    origin_image_tensor = image_rgb_tensor_origin.to(device).type(torch.float)/ 127.5 - 1.0
    pose_tensor = out_pose.to(device).type(torch.float)/ 127.5 - 1.0
    pose_tensor_img= out_pose_origin.to(device).type(torch.float)/ 127.5 - 1.0
    generated_images = gen(origin_image_tensor.unsqueeze(0), pose_tensor.unsqueeze(0), pose_tensor_img.unsqueeze(0))

    pil_image1 = Image.fromarray(((origin_image_tensor+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    tk_image1 = ImageTk.PhotoImage(pil_image1)
    img_label1.config(image=tk_image1)
    img_label1.image = tk_image1

    generated_image_np = ((generated_images[0]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy()
    generated_image_np = generated_image_np  # Rescale to [0, 255]
    generated_image_np = generated_image_np.astype(np.uint8)
    pil_image2 = Image.fromarray(generated_image_np)
    tk_image2 = ImageTk.PhotoImage(pil_image2)
    img_label2.config(image=tk_image2)
    img_label2.image = tk_image2


    pil_image5 = Image.fromarray(image_rgb_tensor.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    tk_image5 = ImageTk.PhotoImage(pil_image5)
    img_label5.config(image=tk_image5)
    img_label5.image = tk_image5
                        

    generated_image_np = out_pose.permute(1,2,0).detach().cpu().numpy()
    generated_image_np = generated_image_np  # Rescale to [0, 255]
    generated_image_np = generated_image_np.astype(np.uint8)
    pil_image6 = Image.fromarray(generated_image_np)
    tk_image6 = ImageTk.PhotoImage(pil_image6)
    img_label6.config(image=tk_image6)
    img_label6.image = tk_image6
                        
    root.update()

