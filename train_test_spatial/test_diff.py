from tqdm import tqdm
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F

sys.path.insert(0, './util')
from utils import save_loss


def test_final_overall(args_final,
                       args_seq,
                       args_diff,
                       trainer,
                       seq_model,
                       data_loader):
        test_final(args_final,
                   args_seq,
                   args_diff,
                   trainer,
                   seq_model,
                   data_loader,
                   None,
                   None)


def test_final(args_final,
               args_seq,
               args_diff,
               trainer,
               model,
               data_loader,
               down_sampler,
               up_sampler,
               save_flag=True):
        contour_dir = os.path.join(args_final.experiment_path, 'contour')
        os.makedirs(contour_dir, exist_ok=True)

        loss_func = torch.nn.MSELoss()
        Nt = args_final.test_Nt
        vf = 1  # video frames per chunk
        Nvf = args_final.test_Nt // vf

        with torch.no_grad():
                print('total ite', len(data_loader))
                for iteration, batch in tqdm(enumerate(data_loader)):
                        batch = batch.to(args_final.device).float()
                        print(f"Original batch shape: {batch.shape}")
                        b_size = batch.shape[0]
                        assert b_size == 1
                        num_time = batch.shape[1]
                        num_velocity = batch.shape[2] if batch.ndim == 5 else 1
                        print(f"num_time: {num_time}, num_velocity: {num_velocity}")
                        H, W = batch.shape[-2], batch.shape[-1]

                        # Same conditioning split as train_diff
                        cutoff = int(H * 0.75)
                        batch_cond_indices = [i for i in range(H) if i % 4 == 0 and i < cutoff]
                        label_indices = [i for i in range(H) if not (i % 4 == 0 and i < cutoff)]
                        original_pred_h = len(label_indices)
                        original_cond_h = len(batch_cond_indices)

                        # Keep unpadded copies for ground truth saving
                        batch_cond_raw = batch[..., batch_cond_indices, :]
                        batch_target_raw = batch[..., label_indices, :]

                        batch_cond = batch[..., batch_cond_indices, :]
                        batch_label = batch[..., label_indices, :]

                        def pad_width_to_16(tensor):
                                h = tensor.shape[-2]
                                pad_h = (16 - h % 16) % 16
                                if pad_h > 0:
                                        B, T, C, H_orig, W_t = tensor.shape
                                        tensor = tensor.reshape(B * T, C, H_orig, W_t)
                                        tensor = F.pad(tensor, (0, 0, 0, pad_h), mode='replicate')
                                        tensor = tensor.reshape(B, T, C, H_orig + pad_h, W_t)
                                return tensor

                        batch_cond = pad_width_to_16(batch_cond)
                        batch_label = pad_width_to_16(batch_label)
                        batch_cond = batch_cond.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

                        seq_name = 'batch' + str(iteration)
                        batch_dir = os.path.join(contour_dir, seq_name)
                        os.makedirs(batch_dir, exist_ok=True)

                        # Sample in chunks of vf frames
                        recon_micro = []
                        for j in range(Nvf):
                                cond_chunk = batch_cond[:, :, vf*j:vf*(j+1), :, :]
                                print(f"cond_chunk shape: {cond_chunk.shape}, video_frames: {vf}")
                                save = trainer.sample(video_frames=vf, cond_images=cond_chunk).detach().cpu().numpy()
                                np.save(os.path.join(batch_dir, f"recon_micro_{j}.npy"), save)
                                np.save(os.path.join(batch_dir, f"recon_micro_{j}gt.npy"),
                                        cond_chunk.detach().cpu().numpy())
                                recon_micro.append(save)

                        # Save ground truth target (label rows, unpadded)
                        gt_target_np = batch_target_raw[0, 0, 0].cpu().numpy()
                        np.save(os.path.join(batch_dir, "ground_truth_target.npy"), gt_target_np)

                        # Save ground truth condition (cond rows, unpadded)
                        gt_cond_np = batch_cond_raw[0, 0, 0].cpu().numpy()
                        np.save(os.path.join(batch_dir, "ground_truth_cond.npy"), gt_cond_np)

                        # Save full original sinogram
                        full_sino_np = batch[0, 0, 0].cpu().numpy()
                        np.save(os.path.join(batch_dir, "ground_truth_full.npy"), full_sino_np)

                        # Save index mappings for SSIM comparison
                        np.save(os.path.join(batch_dir, "cond_indices.npy"), np.array(batch_cond_indices))
                        np.save(os.path.join(batch_dir, "label_indices.npy"), np.array(label_indices))

                        # Concatenate reconstruction chunks and crop padding
                        recon_all = np.concatenate(recon_micro, axis=2)
                        recon_2d = recon_all[0, 0, 0]
                        recon_2d = recon_2d[:original_pred_h, :]

                        # Compute MSE
                        mse = np.mean((recon_2d - gt_target_np) ** 2)
                        print(f"  {seq_name}: MSE = {mse:.6f}")

                        # Visualization
                        if save_flag:
                                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                                axes[0].imshow(full_sino_np, cmap='gray', aspect='auto')
                                axes[0].set_title('Full Sinogram (GT)')
                                axes[0].axis('off')

                                axes[1].imshow(gt_cond_np, cmap='gray', aspect='auto')
                                axes[1].set_title(f'Condition ({len(batch_cond_indices)} rows)')
                                axes[1].axis('off')

                                axes[2].imshow(gt_target_np, cmap='gray', aspect='auto')
                                axes[2].set_title(f'GT Target ({original_pred_h} rows)')
                                axes[2].axis('off')

                                axes[3].imshow(recon_2d, cmap='gray', aspect='auto')
                                axes[3].set_title(f'Reconstruction (MSE={mse:.4f})')
                                axes[3].axis('off')

                                plt.tight_layout()
                                fig.savefig(os.path.join(batch_dir, 'overview.png'),
                                            bbox_inches='tight', dpi=150)
                                plt.close(fig)

        print(f"\nDone! Results saved to {contour_dir}")


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

"""
