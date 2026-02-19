import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import tkinter as tk 
from PIL import Image, ImageTk 
import threading 
import time 
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# from autowu.autowu import AutoWU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# torch.set_float32_matmul_precision("highest") 
# if torch.backends.cudnn.is_available():
#     print(f"cuDNN 사용 가능 여부: {torch.backends.cudnn.is_available()}")
#     print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
# else:
#     print("cuDNN을 사용할 수 없습니다.")
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.deterministic = True 
# torch.backends.cudnn.benchmark = False 
# torch.set_printoptions(profile="full")
root = tk.Tk() 
root.title("train8") 
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
            
        self.pos_encoding1_1 = PositionalEncoding2D(ngf*8, 32, 18)
        self.pos_encoding1_2 = PositionalEncoding2D(ngf*8, 32, 18)

        self.transformer1 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )

        self.pos_encoding2_1 = PositionalEncoding2D(ngf*8, 32, 18)
        self.pos_encoding2_2 = PositionalEncoding2D(ngf*8, 32, 18)

        self.transformer2 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )

        self.pos_encoding3_1 = PositionalEncoding2D(ngf*8, 32, 18)
        self.pos_encoding3_2 = PositionalEncoding2D(ngf*8, 32, 18)

        self.transformer3 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )

        self.pos_encoding4_1 = PositionalEncoding2D(ngf*8, 32, 18)
        self.pos_encoding4_2 = PositionalEncoding2D(ngf*8, 32, 18)

        self.transformer4 = nn.Transformer(
            d_model=ngf*8,
            nhead=nhead,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=True
        )
        
        # self.linear1 = nn.Linear(model_dim, model_dim)
        # self.linear2 = nn.Linear(model_dim, model_dim)
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
    
def thread_def(): 
    batch_number = 25
    save_number = 1
    gen = Generator().to(device) 
    # gen = torch.load("./pth/gen2140001.pt", weights_only=False)
    g_loss_fn = nn.L1Loss() 
    optimizer_g = torch.optim.Adam(gen.parameters(), lr=5e-5)# betas=(0.9, 0.999))
    # scheduler = ReduceLROnPlateau(optimizer_g, "min", factor=0.5, patience=5, min_lr=1e-6)
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False 

   
    while True: 
        for i in range(3417): 
            start_time = time.time() 
            print(f'outdate/origin_img_{i+1}.npy') 
            origin_image_numpy = np.load(f'outdate/origin_img_{i+1}.npy') 
            # pose_numpy = np.load(f'outdate/pose_{i+1}.npy')
            # pose_numpy_img = np.load(f'outdate/pose2_{i+1}.npy') 
            output_image_numpy = np.load(f'outdate/output_img_{i+1}.npy') 
            pose_numpy_output1 = np.load(f'./outdate/pose_img_{i+1}.npy') 
            pose_numpy_output2 = np.load(f'./outdate/pose2_img_{i+1}.npy') 
            origin_image_tensors = torch.from_numpy(origin_image_numpy) 
            # pose_tensors = torch.from_numpy(pose_numpy) 
            # pose_tensors_img = torch.from_numpy(pose_numpy_img) 
            output_image_tensors = torch.from_numpy(output_image_numpy) 
            pose_tensors_output1= torch.from_numpy(pose_numpy_output1) 
            pose_tensors_output2= torch.from_numpy(pose_numpy_output2) 
            end_batch = 0
            for k in range(0, pose_tensors_output1.shape[0], batch_number): 
                save_number +=1 
                if k+batch_number > pose_tensors_output1.shape[0]: 
                    end_batch = pose_tensors_output1.shape[0] 
                else: 
                    end_batch = k+batch_number 
                gen.train() 
                origin_image_tensor = origin_image_tensors[k:end_batch].to(device).type(torch.float)/ 127.5 - 1.0
                # pose_tensor = pose_tensors[k:end_batch].to(device).type(torch.float)/ 127.5 - 1.0
                # pose_tensor_img= pose_tensors_img[k:end_batch].to(device).type(torch.float)/ 127.5 - 1.0
                output_image_tensor = output_image_tensors[k:end_batch].to(device).type(torch.float)/ 127.5 - 1.0
                pose_tensor_output1 = pose_tensors_output1[k:end_batch].to(device).type(torch.float)/ 127.5 - 1.0
                pose_tensor_output2 = pose_tensors_output2[k:end_batch].to(device).type(torch.float)/ 127.5 - 1.0
                
                generated_images = gen(origin_image_tensor, pose_tensor_output1, pose_tensor_output2) 
                g_loss1 = g_loss_fn(generated_images, output_image_tensor.requires_grad_(False)) 
                g_loss = g_loss1 #+ g_loss2 
                optimizer_g.zero_grad()                 
                g_loss.backward() 
                optimizer_g.step()
                # scheduler.step(g_loss)
                end_time = time.time() 
                print(f"실행 시간: {end_time - start_time:.6f} 초 ") 
                print(f'{save_number} g_loss {g_loss}')
                if save_number % 12 == 1: 
                    gen.eval() 
                    with torch.no_grad(): 
                        generated_images_evel = gen(origin_image_tensor[-1:], pose_tensor_output1[-1:], pose_tensor_output2[-1:]) 
                        origin_image_np = ((origin_image_tensor[-1]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy() 
                        origin_image_np = origin_image_np 
                        origin_image_np = origin_image_np.astype(np.uint8) 
                        pil_image1 = Image.fromarray(origin_image_np) 
                        tk_image1 = ImageTk.PhotoImage(pil_image1) 
                        img_label1.config(image=tk_image1) 
                        img_label1.image = tk_image1 
                        output_image_np = ((output_image_tensor[-1]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy() 
                        output_image_np = output_image_np 
                        output_image_np = output_image_np.astype(np.uint8)
                        pil_image2 = Image.fromarray(output_image_np) 
                        tk_image2 = ImageTk.PhotoImage(pil_image2) 
                        img_label2.config(image=tk_image2) 
                        img_label2.image = tk_image2 
                        output_image_np_s = ((generated_images_evel[-1]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy() 
                        output_image_np_s = output_image_np_s 
                        output_image_np_s = output_image_np_s.astype(np.uint8) 
                        pil_image3 = Image.fromarray(output_image_np_s) 
                        tk_image3 = ImageTk.PhotoImage(pil_image3) 
                        img_label3.config(image=tk_image3) 
                        img_label3.image = tk_image3 
                        generated_image_np1 = ((generated_images[-1]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy() 
                        generated_image_np1 = generated_image_np1
                        generated_image_np1 = generated_image_np1.astype(np.uint8) 
                        pil_image4 = Image.fromarray(generated_image_np1) 
                        tk_image4 = ImageTk.PhotoImage(pil_image4) 
                        img_label4.config(image=tk_image4) 
                        img_label4.image = tk_image4 
                        generated_image_np2 = ((pose_tensor_output1[-1]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy() 
                        generated_image_np2 = generated_image_np2 
                        generated_image_np2 = generated_image_np2.astype(np.uint8) 
                        pil_image5 = Image.fromarray(generated_image_np2) 
                        tk_image5 = ImageTk.PhotoImage(pil_image5) 
                        img_label5.config(image=tk_image5) 
                        img_label5.image = tk_image5 
                        generated_image_np3 = ((pose_tensor_output2[-1]+ 1) * 127.5).permute(1,2,0).detach().cpu().numpy() 
                        generated_image_np3 = generated_image_np3 
                        generated_image_np3 = generated_image_np3.astype(np.uint8)
                        pil_image6 = Image.fromarray(generated_image_np3) 
                        tk_image6 = ImageTk.PhotoImage(pil_image6) 
                        img_label6.config(image=tk_image6) 
                        img_label6.image = tk_image6 
                        root.update() 
                        
                if save_number%100000 ==1: 
                    torch.save(gen, f'./pth/train8gen{save_number}.pt') 
def main(): 
    thread1 = threading.Thread(target=thread_def) 
    thread1.start() 
    root.mainloop() 
if "__main__" == __name__:
    main()