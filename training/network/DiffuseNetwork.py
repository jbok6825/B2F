# Diffusion 기반 Facial Animation Generator 전체 구조 (End-to-End 학습)

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.network.module.StyleEncoder import StyleEncoder
from training.network.module.ContentEncoder import ContentEncoder
import torch.nn as nn
import torch
from training.network.Util import *
import torch.optim as optim
from training.network.configs import *
from process_dataset import utils
from diffusers import DDPMScheduler 
import gc


# Sinusoidal timestep embedding
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B]
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  # x: [B, T, D]


class DiffusionTransformerDecoder(nn.Module):
    def __init__(self, device, face_emotion_latent_dim, body_content_latent_dim, face_dim):
        super().__init__()
        self.device = device
        self.latent_dim = face_emotion_latent_dim + body_content_latent_dim
        self.face_dim = face_dim
        self.feature_dim = face_dim

        self.input_proj = nn.Linear(face_dim, self.latent_dim)
        self.style_proj = nn.Linear(face_emotion_latent_dim, self.latent_dim)

        self.null_style_token = nn.Parameter(torch.zeros(1, self.latent_dim))

        self.time_embed = nn.Sequential(
            TimestepEmbedding(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU()
        )

        self.pos_embed = PositionalEncoding(self.latent_dim, max_len=2001)

        self.memory_proj = nn.Linear(body_content_latent_dim + self.latent_dim + self.latent_dim, self.latent_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        self.bs_map_r = nn.Linear(self.latent_dim, face_dim)

    def forward(self, x, t, body_content, style_vector, use_null_style=False):
        B, T, _ = x.shape

        x_proj = self.input_proj(x)  # [B, T, latent_dim]

        if use_null_style:
            style_token = self.null_style_token.expand(B, 1, -1)
        else:
            style_token = self.style_proj(style_vector)[:, None, :]  # [B, 1, latent_dim]

        x_with_style = torch.cat([style_token, x_proj], dim=1)  # [B, T+1, latent_dim]
        x_with_style = self.pos_embed(x_with_style)

        time_emb = self.time_embed(t).to(self.device)[:, None, :].repeat(1, T, 1)
        style_transformed = self.style_proj(style_vector)[:, None, :].repeat(1, T, 1)
        memory_cat = torch.cat([body_content, style_transformed, time_emb], dim=-1)
        memory = self.memory_proj(memory_cat)

        # ⚠️ Fix: Disable tgt_mask for now due to shape mismatch
        decoder_output = self.decoder(x_with_style, memory, tgt_mask=None, memory_mask=None)
        blendshape_output = self.bs_map_r(decoder_output[:, 1:, :])
        return blendshape_output

class FacialDiffusionModel(nn.Module):
    def __init__(self, face_dim, body_dim, face_style_latent_dim, content_latent_dim):
        super().__init__()

        self.styleEncoder = StyleEncoder(input_size=face_dim, hidden_size=128, output_size=face_style_latent_dim, use_vae=False)
        self.bodyContentEncoder = ContentEncoder(body_dim, content_latent_dim, True)
        self.faceContentEncoder = ContentEncoder(56, content_latent_dim, True)
        self.bs_map_content_input = nn.Linear(face_dim, 56)
        self.unet = DiffusionTransformerDecoder(
            device=DEVICE,
            face_emotion_latent_dim=face_style_latent_dim,
            body_content_latent_dim=content_latent_dim,
            face_dim=53
        )

    def forward(self, facial_gt, body_seq, ref_face_seq, noise_scheduler, epoch=None):
        style_vec = self.styleEncoder(ref_face_seq)[0]
        face_content = self.faceContentEncoder(self.bs_map_content_input(facial_gt))
        body_content = self.bodyContentEncoder(body_seq)

        B, T, D = facial_gt.shape
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=facial_gt.device).long()
        noise = torch.randn_like(facial_gt)
        noisy_input = noise_scheduler.add_noise(facial_gt, noise, t)

        prediction = self.unet(noisy_input, t, body_content, style_vec)

        loss_expression = F.mse_loss(prediction[:, :, 3:], facial_gt[:, :, 3:])
        loss_jaw = F.mse_loss(prediction[:, :, :3], facial_gt[:, :, :3])

        ### 🔥 Consistency Loss - 2+3 조합 적용 ###
        # output_face_content = self.faceContentEncoder(self.bs_map_content_input(prediction))
        # output_style_vec = self.styleEncoder(prediction)[0]

        # # 1. t threshold 적용 (작은 t에서만 loss 계산)
        # use_consistency = (t < 200).float().mean() > 0.5  # batch 기준 절반 이상이 low t일 때만 사용

        # # 2. epoch-based weight scheduling
        # if epoch is not None:
        #     base_weight = min(epoch / 30, 1.0)  # 30 epoch까지 선형 증가
        # else:
        #     base_weight = 1.0

        # # raw loss
        # loss_concon_raw = l2distance_loss(face_content, output_face_content)
        # loss_stylecon_raw = l2distance_loss(style_vec, output_style_vec)
        # loss_align_raw = l2distance_loss(face_content, body_content)

        # # mask + weight
        # if use_consistency:
        #     loss_concon = loss_concon_raw * 0.1 * base_weight
        #     loss_stylecon = loss_stylecon_raw * 0.1 * base_weight
        # else:
        #     loss_concon = torch.tensor(0.0, device=facial_gt.device)
        #     loss_stylecon = torch.tensor(0.0, device=facial_gt.device)

        # loss_align = loss_align_raw * 0.1

        # return both
        return {
            'recon': loss_expression + loss_jaw * 1000,
            # 'concon': loss_concon,
            # 'stylecon': loss_stylecon,
            # 'align': loss_align,
            # 'concon_raw': loss_concon_raw,
            # 'stylecon_raw': loss_stylecon_raw,
            # 'align_raw': loss_align_raw,
        }


    @torch.no_grad()
    def sample(self, body_seq, ref_face_seq, noise_scheduler, num_steps=1000):
        B, T, _ = body_seq.shape
        style_vec = self.styleEncoder(ref_face_seq)[0]
        body_content = self.bodyContentEncoder(body_seq)

        x = torch.randn((B, T, self.unet.feature_dim), device=body_seq.device)
        noise_scheduler.set_timesteps(num_steps, device=body_seq.device)

        for t in noise_scheduler.timesteps:
            t_batch = torch.full((B,), t, dtype=torch.long, device=body_seq.device)

            pred_x0 = self.unet(x, t_batch, body_content, style_vec)
            x = noise_scheduler.step(model_output=pred_x0, timestep=t_batch, sample=x).prev_sample

        return x

    def network_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def epoch_process(self, epoch, dataloader, noise_scheduler, writer):
        avg_error_train = 0.
        batch_num = len(dataloader)
        batch_size = dataloader.batch_size

        for training_data in dataloader:
            gc.collect()
            torch.cuda.empty_cache()

            facial_motion_style_0 = training_data['facial_style_feature']
            facial_motion_content_0 = training_data['facial_feature']
            body_motion_content_0 = training_data['fullbody_feature']

            loss = loss = self.forward(
                facial_motion_content_0, 
                body_motion_content_0, 
                facial_motion_style_0, 
                noise_scheduler,
                epoch=epoch  # ⬅️ 여기!
            )

            loss_recon = loss['recon']
            # loss_concon = loss['concon']
            # loss_stylecon = loss['stylecon']
            # loss_align = loss['align']

            total_loss = loss_recon
            
            avg_error_train += total_loss.item() / batch_size
            self.network_update(total_loss)

            writer.add_scalar("loss_total/train", total_loss, epoch)
            writer.add_scalar("loss_recon/train", loss_recon, epoch)

            # ✅ raw 값으로 로그 기록
            # writer.add_scalar("loss_concon/train", loss['concon_raw'], epoch)
            # writer.add_scalar("loss_stylecon/train", loss['stylecon_raw'], epoch)
            # writer.add_scalar("loss_align/train", loss['align_raw'], epoch)


        self.scheduler.step()

        print("epoch: ", epoch, ' avg error_reconstruction:', avg_error_train)
        print('------------------------------------')



    
    def train_network(self, dataloader, epoch_num, writer, save_dir, save_step):
        for param_tensor in self.state_dict():
            self.state_dict()[param_tensor] = self.state_dict()[param_tensor].to(DEVICE)
            self.state_dict()[param_tensor] = self.state_dict()[param_tensor].float()

        self.optimizer = optim.AdamW(self.parameters(),
                                    lr=INIT_LEARNING_RATE, weight_decay=INIT_WEIGHT_DECAY)
        
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="sample") 


        torch.autograd.set_detect_anomaly(True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, Te, Tmult)
        self.criterion = nn.MSELoss()
        self.train()

        save_path = save_dir + "/model"+"_"+"0"+"epoch.pth"
        torch.save(self.state_dict(), save_path)

        for epoch in range(1, epoch_num+1 ):
            self.epoch_process(epoch, dataloader,noise_scheduler, writer)

            if (epoch > 0 and (epoch % save_step == save_step - 1)) or epoch == 1:
                save_path = save_dir + "/model"+"_"+str(epoch)+"epoch.pth"
                torch.save(self, save_path)

        save_path = save_dir + "/model"+"_"+str(epoch)+"epoch.pth"
        torch.save(self, save_path)