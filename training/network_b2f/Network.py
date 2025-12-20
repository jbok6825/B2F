from training.network_b2f.module.StyleEncoder import StyleEncoder
from training.network_b2f.module.ContentEncoder import ContentEncoder
import torch.nn as nn
import torch
from training.network_b2f.utils import *
from training.network_b2f.module.TransformerDecoder import TransformerDecoder
import gc
from training.network_b2f.configs import *
from process_dataset import utils

import torch.optim as optim

class Network(nn.Module):
    def __init__(self, 
                 device,  
                 face_dim,
                 body_dim,
                 face_emotion_latent_dim,
                 face_content_latent_dim,
                 body_content_latent_dim,
                 same_mode = False,
                 use_normalize = True,
                 vae_mode = True,
                 content_vae_mode = False,
                 positional_encoding = True,  # decoder에 positional encoding 안 씀
                 ):
        super(Network, self).__init__()
        self.device = device
        self.same_mode = same_mode
        self.vae_mode = vae_mode
        self.content_vae_mode = content_vae_mode

        self.use_body_content_encoder = not same_mode

        self.emotionEncoder = StyleEncoder(
            input_size=face_dim, hidden_size=256,
            output_size=face_emotion_latent_dim,
            use_vae=vae_mode, use_normalize=use_normalize
        )

        self.faceContentEncoder = ContentEncoder(56, face_content_latent_dim, positional_encoding, use_vqvae=content_vae_mode)
        
        if self.use_body_content_encoder:
            self.bodyContentEncoder = ContentEncoder(body_dim, body_content_latent_dim, positional_encoding, use_vqvae=content_vae_mode)
        else:

            self.body_input_proj = nn.Linear(body_dim, body_content_latent_dim)

        self.memory_projection_content = nn.Linear(body_content_latent_dim, body_content_latent_dim+face_emotion_latent_dim)

        self.decoder = TransformerDecoder(
            device=device,
            feature_dim=face_emotion_latent_dim + body_content_latent_dim
        )


        self.bs_map_r = nn.Linear(face_emotion_latent_dim + body_content_latent_dim, face_dim)
        self.bs_map_content_input = nn.Linear(face_dim, 56)

        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    def forward(self, input, is_runtime=False, style_embedding=None):
        input_body = input['body_motion_content']

        mu = out_mu = logvar = None
        quantized_body = projected_body = None
        quantized_face = projected_face = None

        if self.content_vae_mode == True:
            bodyContentEmbedding, quantized_body, projected_body = self.bodyContentEncoder(input_body)
        else:
            if self.use_body_content_encoder:
                bodyContentEmbedding = self.bodyContentEncoder(input_body)
            else:
                bodyContentEmbedding = self.body_input_proj(input_body)
        frame_num = bodyContentEmbedding.shape[1]


        


        if style_embedding is None:
            input_emotion = input['facial_motion_style']
            if self.vae_mode:
                faceStyleEmbedding, mu, logvar = self.emotionEncoder(input_emotion)
                
            else:
                faceStyleEmbedding = self.emotionEncoder(input_emotion)[0]
        else:

            faceStyleEmbedding = style_embedding


        # style 변환 및 broadcast
        style_proj = faceStyleEmbedding  # [B, style_dim]
        style_broadcast = style_proj.unsqueeze(1).repeat(1, frame_num, 1)  # [B, T, style_dim]


        # decoder input (tgt): style + content
        tgt = torch.cat([style_broadcast, bodyContentEmbedding], dim=-1).transpose(0, 1)  # [T, B, full_dim]

        # decoder memory: content only
        memory = self.memory_projection_content(bodyContentEmbedding).transpose(0, 1)  # [T, B, content_dim]

    
        # decoding
        decoder_output = self.decoder(tgt, memory).transpose(0, 1)  # [B, T, full_dim]
        blendshape_output = self.bs_map_r(decoder_output)  # [B, T, face_dim]

        if not is_runtime:
            input_faceContentEncoder = input['facial_motion_content']
            if self.content_vae_mode == True:
                faceContentEmbedding, quantized_face, projected_face = self.faceContentEncoder(self.bs_map_content_input(input_faceContentEncoder))
                outputFaceContentEmbedding, _, _ = self.faceContentEncoder(self.bs_map_content_input(blendshape_output))
            else:
                faceContentEmbedding = self.faceContentEncoder(self.bs_map_content_input(input_faceContentEncoder))
                outputFaceContentEmbedding = self.faceContentEncoder(self.bs_map_content_input(blendshape_output))

            outputFaceStyleEmbedding, out_mu, out_logvar = self.emotionEncoder(blendshape_output)
            return {
                'blendshape_output': blendshape_output,
                'faceStyleEmbedding': faceStyleEmbedding,
                'faceContentEmbedding': faceContentEmbedding,
                'bodyContentEmbedding': bodyContentEmbedding,
                'outputFaceStyleEmbedding': outputFaceStyleEmbedding,
                'outputFaceContentEmbedding': outputFaceContentEmbedding,
                'mu': mu,
                'out_mu': out_mu,
                'logvar': logvar,
                'quantized_body': quantized_body,
                'projected_body': projected_body,
                'quantized_face': quantized_face,
                'projected_face': projected_face,
            }

        else:
            return {
                'blendshape_output': blendshape_output,
                'faceStyleEmbedding': faceStyleEmbedding,
                'mu': mu,
                'logvar': logvar
            }


    # def forward_with_two_style(self, input, smooth_interpolation = False, ratio = None):
        
    #     input_bodyContentEncoder = input['body_motion_content']

    #     bodyContentEmbedding = self.bodyContentEncoder(input_bodyContentEncoder)



    #     frame_num = bodyContentEmbedding.shape[1]


    #     input_emotionEncoder_0 = input['facial_motion_style_0']
    #     input_emotionEncoder_1 = input['facial_motion_style_1']
    #     faceStyleEmbedding_0, logits_0, _ = self.emotionEncoder(input_emotionEncoder_0)
    #     faceStyleEmbedding_1,logits_1, _  = self.emotionEncoder(input_emotionEncoder_1)


    #     if smooth_interpolation == True:


    #         # 1. 프레임별 알파 생성 (0 → 1로 선형 증가) : shape [T]
    #         alpha = torch.linspace(0, 1, steps=frame_num, device=faceStyleEmbedding_0.device)  # [T]

    #         # 2. 배치 차원과 맞추기 위해 shape 변경: [1, T, 1]
    #         alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]  
    #         # => 나중에 [B, T, style_dim]과 브로드캐스트 가능

    #         # 3. 각 style을 [B, 1, style_dim] → [B, T, style_dim]으로 broadcast
    #         style0 = faceStyleEmbedding_0.unsqueeze(1).expand(-1, frame_num, -1)  # [B, T, style_dim]
    #         style1 = faceStyleEmbedding_1.unsqueeze(1).expand(-1, frame_num, -1)  # [B, T, style_dim]

    #         # 4. 프레임에 따라 alpha로 interpolation
    #         style_broadcast = (1 - alpha) * style1 + alpha * style0



    #         # decoder input (tgt): style + content
    #         tgt = torch.cat([style_broadcast, bodyContentEmbedding], dim=-1).transpose(0, 1)  # [T, B, full_dim]

    #         # decoder memory: content only
    #         memory = self.memory_projection_content(bodyContentEmbedding).transpose(0, 1)  # [T, B, content_dim]

        
    #         # decoding
    #         decoder_output = self.decoder(tgt, memory).transpose(0, 1)  # [B, T, full_dim]
    #         blendshape_output = self.bs_map_r(decoder_output)  # [B, T, face_dim]

    #         return {
    #             'blendshape_output': blendshape_output,
    #             'faceStyleEmbedding': None,
    #             'mu': None,
    #             'logvar': None
    #         }



    #     else:
    #         # new_logit = (1-ratio) * logits_0 + ratio * logits_1
    #         # new_styleembedding = self.emotionEncoder.forward(logits=new_logit)[0]
    #         new_styleembedding = (1 - ratio) * faceStyleEmbedding_0 + ratio * faceStyleEmbedding_1  # [B, style_dim]

    #         # 시간 축으로 broadcast
    #         style_broadcast = new_styleembedding.unsqueeze(1).repeat(1, frame_num, 1) 
    #         # decoder input (tgt): style + content
    #         tgt = torch.cat([style_broadcast, bodyContentEmbedding], dim=-1).transpose(0, 1)  # [T, B, full_dim]

    #         # decoder memory: content only
    #         memory = self.memory_projection_content(bodyContentEmbedding).transpose(0, 1)  # [T, B, content_dim]

        
    #         # decoding
    #         decoder_output = self.decoder(tgt, memory).transpose(0, 1)  # [B, T, full_dim]
    #         blendshape_output = self.bs_map_r(decoder_output)  # [B, T, face_dim]

    #         return {
    #             'blendshape_output': blendshape_output,
    #             'faceStyleEmbedding': None,
    #             'mu': None,
    #             'logvar': None
    #         }



        
        


    def loss_reconstruction_calculate(self, prediction, original_data, jaw_weight):
        loss_expression = F.mse_loss(prediction[:, :, 3:], original_data[:, :, 3:])
        loss_jaw = F.mse_loss(prediction[:, :, :3], original_data[:, :, :3])
        
        loss_reconstruction = loss_expression + loss_jaw  *jaw_weight

        return loss_reconstruction
    



    
    def loss_without_reconstruction_calculate(self, output, style_code):
        # if self.vae_mode == True:
        #     faceStyleEmbedding = output['mu']
        #     outputFaceStyleEmbedding = output['out_mu']
        # else:
        faceStyleEmbedding = output['faceStyleEmbedding']
        outputFaceStyleEmbedding = output['outputFaceStyleEmbedding']
        
        faceContentEmbedding = output['faceContentEmbedding']
        bodyContentEmbedding = output['bodyContentEmbedding']
        outputFaceContentEmbedding = output['outputFaceContentEmbedding']
        

        # loss_align = l2distance_loss(faceContentEmbedding, bodyContentEmbedding)
        loss_align = 1 - F.cosine_similarity(faceContentEmbedding, bodyContentEmbedding, dim=-1).mean()
        loss_content = l2distance_loss(faceContentEmbedding, outputFaceContentEmbedding)
        loss_style = l2distance_loss(faceStyleEmbedding, outputFaceStyleEmbedding)


        return loss_align, loss_content, loss_style



    def network_update(self, loss):
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
    
 
    def epoch_process_step4_origin(self, epoch, dataloader, dataloader_sub, writer, jaw_weight = 1000, kl_weight = 0.5):
        print("@@Epoch:", epoch)
        avg_error_train = 0.
        batch_num = len(dataloader)
        batch_size = dataloader.batch_size
        

        # epcoh 2개 단위로 들어온다고 
        
        for training_data_main, training_data_sub in zip(dataloader, dataloader_sub):
            gc.collect()
            torch.cuda.empty_cache()
            training_data = [training_data_main, training_data_sub]

            facial_motion_style_0 = training_data[0]['facial_style_feature']
            facial_motion_content_0 = training_data[0]['facial_feature']
            body_motion_content_0 = training_data[0]['fullbody_feature']
            style_code_0 = training_data[0]['style_code']

            facial_motion_style_1 = training_data[1]['facial_style_feature']
            facial_motion_content_1 = training_data[1]['facial_feature']
            body_motion_content_1 = training_data[1]['fullbody_feature']
            style_code_1 = training_data[1]['style_code']

            # first step (reconstruction loss 0)
            output = self.forward({
                'facial_motion_style': facial_motion_style_0,
                'facial_motion_content': facial_motion_content_0,
                'body_motion_content':body_motion_content_0
            })

            loss_reconstruction_step1 = self.loss_reconstruction_calculate(output['blendshape_output'], facial_motion_content_0, jaw_weight)

            loss_align_step1, loss_content_step1, loss_style_step1 =  self.loss_without_reconstruction_calculate(output, style_code_0)

        
            loss_total = loss_reconstruction_step1 * 5.0 + loss_align_step1 * 0.5  + loss_content_step1 * 0.5  + loss_style_step1 * 1.0

            if self.vae_mode == True:
                mu = output['mu']
                kl_loss = self.emotionEncoder.kl_gumbel_softmax_uniform(mu)
                loss_kl_step1 = kl_loss * kl_weight
                loss_total = loss_total + loss_kl_step1
                # writer.add_scalar("loss_kl/train", kl_loss.item(), epoch)


            writer.add_scalar("loss_reconstruction/train", loss_reconstruction_step1, epoch)
            writer.add_scalar("loss_align/train", loss_align_step1, epoch)
            writer.add_scalar("loss_style/train", loss_style_step1, epoch)
            writer.add_scalar("loss_content/train", loss_content_step1, epoch)
 
           
            
            

            # second step (style 1 + content 0)

            output = self.forward({
                'facial_motion_style': facial_motion_style_1,
                'facial_motion_content': facial_motion_content_0,
                'body_motion_content':body_motion_content_0
            })

            loss_align_step2, loss_content_step2, loss_style_step2 = self.loss_without_reconstruction_calculate(output, style_code_1)
            loss_total = loss_total + loss_content_step2 *0.1  + loss_style_step2 *0.5



            writer.add_scalar("cross_style/train", loss_style_step2, epoch)
            writer.add_scalar("cross_content/train", loss_content_step2, epoch)

            
            writer.add_scalar("Total Loss/train",loss_total, epoch)
            self.network_update(loss_total)
            avg_error_train += (loss_total).item() / batch_num

            # third step (reconstruction 1)

            output = self.forward({
                'facial_motion_style': facial_motion_style_1,
                'facial_motion_content': facial_motion_content_1,
                'body_motion_content':body_motion_content_1
            })

            loss_reconstruction_step3 = self.loss_reconstruction_calculate(output['blendshape_output'], facial_motion_content_1, jaw_weight)
            loss_align_step3, loss_content_step3, loss_style_step3 = self.loss_without_reconstruction_calculate(output, style_code_1)
            

            loss_total = loss_reconstruction_step3 * 5.0 + loss_align_step3 * 0.5 + loss_content_step3 * 0.5  + loss_style_step3 *1.0
            
            if self.vae_mode == True:
                mu = output['mu']

                kl_loss = self.emotionEncoder.kl_gumbel_softmax_uniform(mu)
                loss_kl_step3 = kl_loss * kl_weight
                loss_total = loss_total + loss_kl_step3
                writer.add_scalar("loss_kl/train", kl_loss.item(), epoch+1)




            writer.add_scalar("loss_reconstruction/train", loss_reconstruction_step3, epoch+1)
            writer.add_scalar("loss_align/train", loss_align_step3, epoch+1)
            writer.add_scalar("loss_style/train", loss_style_step3, epoch+1)
            writer.add_scalar("loss_content/train", loss_content_step3, epoch+1)
            
  

            # fourth step
            output = self.forward({
                'facial_motion_style': facial_motion_style_0,
                'facial_motion_content': facial_motion_content_1,
                'body_motion_content':body_motion_content_1
            })

            loss_align_step4, loss_content_step4, loss_style_step4 = self.loss_without_reconstruction_calculate(output, style_code_0)
            loss_total = loss_total + loss_content_step4 *0.1  + loss_style_step4 *0.5



            writer.add_scalar("cross_style/train", loss_style_step4, epoch+1)
            writer.add_scalar("cross_content/train", loss_content_step4, epoch+1)

            
            writer.add_scalar("Total Loss/train",loss_total, epoch)
            self.network_update(loss_total)
            avg_error_train += (loss_total).item() / batch_num


        self.scheduler.step()

        print('avg error_reconstruction:', avg_error_train)
        print('------------------------------------')

    




    def train_network(self, dataloader, epoch_num, writer, save_dir, save_step, dataloader_sub = None, mode = "origin"):
        for param_tensor in self.state_dict():
            self.state_dict()[param_tensor] = self.state_dict()[param_tensor].to(self.device)
            self.state_dict()[param_tensor] = self.state_dict()[param_tensor].float()

        self.optimizer = optim.AdamW(self.parameters(),
                                    lr=INIT_LEARNING_RATE, weight_decay=INIT_WEIGHT_DECAY)
        
        

        torch.autograd.set_detect_anomaly(True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, Te, Tmult)
        self.criterion = nn.MSELoss()
        self.train()

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} requires grad")
            
        save_path = save_dir + "/model"+"_"+"0"+"epoch.pth"
        torch.save(self, save_path)


        if mode == "origin":

            for epoch in range(1, epoch_num+1, 2):
                kl_weight = compute_kl_weight(epoch, epoch_num+1)
                self.epoch_process_step4_origin(epoch, dataloader, dataloader_sub, writer, kl_weight)

                if (epoch > 0 and (epoch % save_step == save_step - 1 or epoch % save_step == save_step-2)) or epoch == 1:
                    save_path = save_dir + "/model"+"_"+str(epoch)+"epoch.pth"
                    torch.save(self, save_path)

        save_path = save_dir + "/model"+"_"+str(epoch_num)+"epoch.pth"
        torch.save(self, save_path)
        
        writer.flush()
        writer.close()
