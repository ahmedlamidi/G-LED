from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import torch.nn.functional as F

sys.path.insert(0, './util')
from utils import save_loss

def test_final_overall(args_final, 
                                           args_seq, 
                                           args_diff, 
                                           trainer, 
                                           seq_model,
                                           data_loader):
        # down_sampler = torch.nn.Upsample(size=args_seq.coarse_dim, 
        #                                                                mode=args_seq.coarse_mode)
        # up_sampler   = torch.nn.Upsample(size=[512, 512], 
        #                                                                mode=args_seq.coarse_mode)
        #max_mre,min_mre, mean_mre, sigma3 = 
        test_final(args_final, 
                           args_seq, 
                           args_diff, 
                           trainer, 
                           seq_model, 
                           data_loader,
                           None,
                           None)
        #print('#### max mre####=',max_mre)
        #print('#### mean mre####=',mean_mre)
        #print('#### min mre####=',min_mre)
        #print('#### 3 sigma ####=',sigma3)
        pdb.set_trace()

def test_final(args_final, 
                           args_seq, 
                           args_diff, 
                           trainer, 
                           model, 
                           data_loader,
                           down_sampler,
                           up_sampler,
                           save_flag=True):
        try:
                os.makedirs(args_final.experiment_path+'/contour')
        except:
                pass
        contour_dir = args_final.experiment_path+'/contour'
        loss_func = torch.nn.MSELoss()
        Nt = args_final.test_Nt
        LED_micro_list = []
        LED_macro_list = []
        Dif_recon_list = []
        CFD_micro_list = []
        CFD_macro_list = []
        with torch.no_grad():
                #IDHistory = [i for i in range(1, args_seq.n_ctx)]
                #REs = []
                #REs_fine = []
                print('total ite', len(data_loader))
                for iteration, batch in tqdm(enumerate(data_loader)):
                        batch = batch.to(args_final.device).float()
                        print(f"Original batch shape: {batch.shape}")  # Debug: see actual data shape
                        b_size = batch.shape[0]
                        assert b_size == 1
                        num_time = batch.shape[1]
                        num_velocity = batch.shape[2] if batch.ndim == 5 else 1  # Get actual channels from data
                        print(f"num_time: {num_time}, num_velocity: {num_velocity}")
                        H, W = batch.shape[-2], batch.shape[-1]
                        half = H // 2
                        cond_indices = [i for i in range(0, half, 4)] + [i for i in range(half + 2, H, 4)]
                        cond_set = set(cond_indices)
                        label_indices = [i for i in range(H) if i not in cond_set]
                        batch_cond = batch[..., cond_indices, :]  # Asymmetric conjugate condition
                        batch      = batch[..., label_indices, :]  # Remaining levels - to predict
                        def pad_width_to_16(tensor):
                                # tensor shape: [B, T, C, H, W]
                                h = tensor.shape[-2]
                                pad_h = (16 - h % 16) % 16
                                if pad_h > 0:
                                        # Reshape to 4D for padding (F.pad reflect doesn't support 5D)
                                        B, T, C, H_orig, W_t = tensor.shape
                                        tensor = tensor.reshape(B * T, C, H_orig, W_t)
                                        # For 4D tensor: (left, right, top, bottom)
                                        tensor = F.pad(tensor, (0, 0, 0, pad_h), mode='reflect')
                                        tensor = tensor.reshape(B, T, C, H_orig + pad_h, W_t)
                                return tensor

                        batch_cond = pad_width_to_16(batch_cond)
                        batch = pad_width_to_16(batch)
                        batch_cond = batch_cond.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                        recon_micro = []
                        vf = 1 #args_diff.nt
                        Nvf = args_final.test_Nt // vf

    
                        seq_name = 'batch'+str(iteration)
                        try:
                            os.makedirs(contour_dir+'/'+seq_name)
                        except:
                            pass
    
			# Sample in chunks of vf frames
                        recon_micro = []
                        for j in range(Nvf):
				# Slice along time dimension (dim 2): batch_cond is [B, C, T, H, W]
				# Need exactly vf frames to match video_frames parameter
                                cond_chunk = batch_cond[:, :, vf*j:vf*(j+1), :, :]
                                print(f"cond_chunk shape: {cond_chunk.shape}, video_frames: {vf}")
                                save = trainer.sample(video_frames=vf, cond_images=cond_chunk).detach().cpu().numpy()
                                np.save(contour_dir+'/'+seq_name+"/recon_micro_"+str(j)+".npy", save)
                                np.save(contour_dir+'/'+seq_name+"/recon_micro_"+str(j)+"gt"+".npy", cond_chunk.detach().cpu().numpy())
                                recon_micro.append(save)
			#recon_micro = torch.cat(recon_micro, dim=2)  # concatenate along time dimension

			#DIC = {"recon_micro":recon_micro.detach().cpu().numpy()}
			#pickle.dump(DIC, open(contour_dir+'/'+seq_name+"/DIC.pkl", 'wb'), protocol=4)
			# Also save as raw numpy

                                # data is nn prediction
                        #       data = mem[i].cpu().numpy()
                        #       X_AR = batch_coarse_merge_spatial[i,previous_len:previous_len+Nt+1,:].cpu().numpy()
                        #       data = data.reshape([-1,16])
                        #       X_AR = X_AR.reshape([-1,16])    
                        #       pdb.set_trace()

                        #       mem  = mem.reshape([mem.shape[0],
                        #                                                                       1,
                        #                                                                       mem.shape[1],
                        #                                                                       mem.shape[2]])
                        # mem2fine = []
                        # for nc in range(int(mem.shape[2]/args_diff.Nt)):
                        #       mem2fine.append(up_sampler(mem[:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])) 
                        # mem2fine = torch.cat(mem2fine,dim=2)
                        
                        # mem2fine = mem2fine.reshape([mem2fine.shape[0],
                        #                                                               1,
                        #                                                               1,
                        #                                                               mem2fine.shape[-2],
                        #                                                               mem2fine.shape[-1]])




"""
def test_final(args_final, 
                           args_seq, 
                           args_diff, 
                           trainer, 
                           model, 
                           data_loader,
                           down_sampler,up_sampler):
        print('Iteration is ', len(data_loader))
        IDHistory = [i for i in range(1, args_seq.n_ctx)]
        with torch.no_grad():
                for iteration, batch in tqdm(enumerate(data_loader)):   
                        batch_coarse = down_sampler(batch)
                        bcfs = [batch_coarse.shape[0],batch_coarse.shape[1],args_seq.coarse_dim[0]*args_seq.coarse_dim[1]]
                        batch_coarse_flatten = batch_coarse.reshape(bcfs)

                        


                        
                        bffs = [batch.shape[0], batch.shape[1], 64]
                        batch_fine_flatten   = batch.reshape(bffs)

                        len_batch = batch_fine_flatten.shape[1]

                        coarse_one  = batch_coarse_flatten[:,:args_final.test_Nt,:]

                        if args_final.warm_up:
                                _,past,_,_=model(inputs_embeds = coarse_one[:,0:args_seq.n_ctx-1,:], past=None)
                                xn = coarse_one[:,args_seq.n_ctx:args_seq.n_ctx+1,:]
                                previous_len = args_seq.n_ctx
                        else:
                                past = None
                                xn = coarse_one[:,0:1,:]
                                previous_len = 1 
                        mem = []
                        for j in tqdm(range(args_final.test_Nt-1)):
                                if j == 0:
                                        xnp1,past,_,_=model(inputs_embeds = xn, past=past)
                                elif past[0][0].shape[2] < args_seq.n_ctx and j > 0:
                                        if args_final.warm_up:
                                                raise ValueError("Should not stop here")
                                        xnp1,past,_,_=model(inputs_embeds = xn, past=past)
                                else:
                                        past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args_seq.n_layer)]
                                        xnp1,past,_,_=model(inputs_embeds = xn, past=past)
                                xn = xnp1
                                mem.append(xn)
                        mem=torch.cat([coarse_one[:,0:1,:]]+mem,dim=1)

                        


                        mem  = mem.reshape([mem.shape[0],
                                                                                                1,
                                                                                                mem.shape[1],
                                                                                                mem.shape[2]])
                        mem2fine = []
                        for nc in range(int(mem.shape[2]/args_diff.Nt)):
                                mem2fine.append(up_sampler(mem[:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])) 
                        mem2fine = torch.cat(mem2fine,dim=2)
                        
                        mem2fine = mem2fine.reshape([mem2fine.shape[0],
                                                                                        1,
                                                                                        1,
                                                                                        mem2fine.shape[-2],
                                                                                        mem2fine.shape[-1]])

                        coarse_one  = coarse_one.reshape([coarse_one.shape[0],
                                                                                                1,
                                                                                                coarse_one.shape[1],
                                                                                                coarse_one.shape[2]])
                        
                        coarse2fine = []

                        for nc in range(int(coarse_one.shape[2]/args_diff.Nt)):
                                coarse2fine.append(up_sampler(coarse_one[:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])) 
                        coarse2fine = torch.cat(coarse2fine,dim=2)
                        
                        
                        fine_one    = batch_fine_flatten[:,:args_final.test_Nt]
                        fine_one    = fine_one.reshape([fine_one.shape[0],
                                                                                        1,
                                                                                        1,
                                                                                        fine_one.shape[1],
                                                                                        fine_one.shape[2]])
                        
                        coarse2fine = coarse2fine.reshape([fine_one.shape[0],
                                                                                        1,
                                                                                        1,
                                                                                        fine_one.shape[-2],
                                                                                        fine_one.shape[-1]])
                        data = []
                        data_led = []
                        for nc in tqdm(range(int(coarse_one.shape[2]/args_diff.Nt))):
                                # B x T x F x H x W
                                les_video_sampled_chunck = trainer.sample(video_frames=1, 
                                                                          cond_images=coarse2fine[:,:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])
                                les_video_sampled_chunck_led = trainer.sample(video_frames=1, 
                                                                          cond_images=mem2fine[:,:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])
                                data.append(les_video_sampled_chunck[0,0,0].cpu().numpy())
                                data_led.append(les_video_sampled_chunck_led[0,0,0].cpu().numpy())
                        data = np.vstack(data)
                        data_led = np.vstack(data_led)
                        X_AR = fine_one[0,0,0].cpu().numpy()
                        coarse2fine_np = coarse2fine[0,0,0].cpu().numpy()
                        mem2fine_np    = mem2fine[0,0,0].detach().cpu().numpy() 

                        asp = data.shape[1]/data.shape[0]*3
                        fig, axes = plt.subplots(nrows=1, ncols=5)
                        norm = matplotlib.colors.Normalize(vmin=X_AR.min(), vmax=X_AR.max())
                        im0 = axes[0].imshow(data[:,:],
                                                                aspect=asp,cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[0].title.set_text('micro \n reconstruction')
                        axes[0].invert_yaxis()
                        axes[0].set_xlabel('x')
                        axes[0].set_ylabel('t')
                        axes[0].set_xticks([])
                        axes[0].set_yticks([])

                        im1 = axes[1].imshow(data_led[:,:],
                                                                aspect=asp,cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[1].title.set_text('micro \n LED')
                        axes[1].invert_yaxis()
                        axes[1].set_xlabel('x')
                        axes[1].set_xticks([])
                        axes[1].set_yticks([])
                        
                        
                        im2 = axes[2].imshow(X_AR[:,:],aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[2].title.set_text('micro \n truth')
                        axes[2].invert_yaxis()
                        axes[2].set_xlabel('x')
                        axes[2].set_yticks([])
                        axes[2].set_xticks([])
                        
                        # im2 = axes[3].imshow(np.abs(X_AR-data),aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        # axes[3].title.set_text('En/De- \n Coder \n Error')
                        # axes[3].invert_yaxis()
                        # axes[3].set_xlabel('x')
                        # axes[3].set_yticks([])

                        im3 = axes[3].imshow(coarse2fine_np,aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[3].title.set_text('macro \n truth')
                        axes[3].invert_yaxis()
                        axes[3].set_xlabel('x')
                        axes[3].set_yticks([])
                        axes[3].set_xticks([])

                        im4 = axes[4].imshow(mem2fine_np,aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[4].title.set_text('macro \n LED')
                        axes[4].invert_yaxis()
                        axes[4].set_xlabel('x')
                        axes[4].set_yticks([])
                        axes[4].set_xticks([])
                        
                        fig.subplots_adjust(right=0.8)
                        fig.colorbar(im0,orientation="horizontal",ax = axes)
                        fig.savefig('./batch'+str(iteration)+'sample_kas'+'Nt_read'+str(args_final.Nt_read)+'.png', bbox_inches='tight',dpi=500)
                        plt.close(fig)


                        fig, axes = plt.subplots(nrows=1, ncols=4)
                        norm = matplotlib.colors.Normalize(vmin=X_AR.min(), vmax=X_AR.max())
                        im0 = axes[0].imshow(data[:,:],
                                                                aspect=asp,cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[0].title.set_text('decoder \n reconstruction')
                        axes[0].invert_yaxis()
                        axes[0].set_xlabel('x')
                        axes[0].set_ylabel('t')
                        axes[0].set_xticks([])
                        axes[0].set_yticks([])

                        
                        
                        im2 = axes[2].imshow(X_AR[:,:],aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[2].title.set_text('micro \n truth')
                        axes[2].invert_yaxis()
                        axes[2].set_xlabel('x')
                        axes[2].set_yticks([])
                        axes[2].set_xticks([])
                        
                        # im2 = axes[3].imshow(np.abs(X_AR-data),aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        # axes[3].title.set_text('En/De- \n Coder \n Error')
                        # axes[3].invert_yaxis()
                        # axes[3].set_xlabel('x')
                        # axes[3].set_yticks([])

                        im3 = axes[1].imshow(coarse2fine_np,aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[1].title.set_text('encoder \n output')
                        axes[1].invert_yaxis()
                        axes[1].set_xlabel('x')
                        axes[1].set_yticks([])
                        axes[1].set_xticks([])

                        im4 = axes[3].imshow(np.abs(data[:,:]-X_AR[:,:]),aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
                        axes[3].title.set_text('En/De-coder \n error')
                        axes[3].invert_yaxis()
                        axes[3].set_xlabel('x')
                        axes[3].set_yticks([])
                        axes[3].set_xticks([])
                        
                        fig.subplots_adjust(right=0.8)
                        fig.colorbar(im0,orientation="horizontal",ax = axes)
                        fig.savefig('./batch'+str(iteration)+'sample_kas'+'Nt_read'+str(args_final.Nt_read)+'endecoder.png', bbox_inches='tight',dpi=500)
                        plt.close(fig)
                        pdb.set_trace()

"""                     
