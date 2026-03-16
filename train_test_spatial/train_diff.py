from tqdm import tqdm
import torch
import torch.nn.functional as F
import pdb
import sys
import os
import numpy as np
sys.path.insert(0, './util')
from utils import save_loss

def train_diff(diff_args,
			   seq_args,
			   trainer,
			   data_loader,
			   start_epoch=0):
	# Load previous loss lists if resuming (support both old and new format)
	total_loss_list, data_loss_list, physics_loss_list = [], [], []
	
	# Try loading new format first
	total_file = os.path.join(diff_args.logging_path, 'total_loss.txt')
	data_file = os.path.join(diff_args.logging_path, 'data_loss.txt')
	physics_file = os.path.join(diff_args.logging_path, 'physics_loss.txt')
	
	if start_epoch > 0:
		if os.path.exists(total_file):
			total_loss_list = list(np.loadtxt(total_file))
			data_loss_list = list(np.loadtxt(data_file)) if os.path.exists(data_file) else []
			physics_loss_list = list(np.loadtxt(physics_file)) if os.path.exists(physics_file) else []
			print(f"Loaded {len(total_loss_list)} previous loss values")
		else:
			# Fallback to old format
			old_loss_file = os.path.join(diff_args.logging_path, 'loss_curve.txt')
			if os.path.exists(old_loss_file):
				total_loss_list = list(np.loadtxt(old_loss_file))
				print(f"Loaded {len(total_loss_list)} previous loss values (old format)")
	
	# Try to get start_epoch from saved checkpoint epoch file
	epoch_file = os.path.join(diff_args.model_save_path, 'best_model_sofar_epoch')
	if start_epoch == 0 and os.path.exists(epoch_file):
		saved_epoch = int(np.loadtxt(epoch_file)[0])
		if hasattr(diff_args, 'resume') and diff_args.resume:
			start_epoch = saved_epoch + 1
			print(f"Resuming from epoch {start_epoch}")
	
	for epoch in range(start_epoch, diff_args.epoch_num):
		down_sampler = torch.nn.Upsample(size=seq_args.coarse_dim, 
								     	 mode=seq_args.coarse_mode)
		up_sampler   = torch.nn.Upsample(size=[720, 448], 
								     	 mode=seq_args.coarse_mode)
		model, loss_components = train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler)
		total_loss, data_loss, physics_loss = loss_components
		
		if epoch % 1 ==0 and epoch > 0:
			peep = 0
			#save_loss(diff_args, loss_list+[loss],epoch)
			#model.save(path=os.path.join(diff_args.model_save_path, 
			#							 'model_epoch_' + str(epoch)))
		
		# Save checkpoint every 10 epochs for resumability
		if epoch > 0 and epoch % 100 == 0:
			model.save(path=os.path.join(diff_args.model_save_path, 
										'checkpoint_epoch_' + str(epoch)))
			print(f"Saved checkpoint at epoch {epoch}")
		
		# Save latest checkpoint (overwritten each epoch for quick resume)
		model.save(path=os.path.join(diff_args.model_save_path, 'latest_checkpoint'))
		np.savetxt(os.path.join(diff_args.model_save_path, 'latest_epoch'), np.array([epoch]))
		
		# Update loss lists
		total_loss_list.append(total_loss)
		data_loss_list.append(data_loss) 
		physics_loss_list.append(physics_loss)
		
		# Save loss values every epoch
		np.savetxt(os.path.join(diff_args.logging_path, 'total_loss.txt'), total_loss_list)
		np.savetxt(os.path.join(diff_args.logging_path, 'data_loss.txt'), data_loss_list)
		np.savetxt(os.path.join(diff_args.logging_path, 'physics_loss.txt'), physics_loss_list)
		
		# Save best model when total loss improves
		if epoch >= 1 and total_loss < min(total_loss_list[:-1], default=float('inf')):
			model.save(path=os.path.join(diff_args.model_save_path, 
										'best_model_sofar'))
			np.savetxt(os.path.join(diff_args.model_save_path, 
								'best_model_sofar_epoch'),np.ones(2)*epoch)
		
		print(f"Epoch {epoch}: Total Loss={total_loss:.6f}, Data Loss={data_loss:.6f}, Physics Loss={physics_loss:.6f}")


def train_epoch(diff_args, seq_args, trainer, data_loader, down_sampler, up_sampler):
    loss_epoch = []
    data_loss_epoch = []
    physics_loss_epoch = []

    # Pre-compute condition indices once (constant across all iterations)
    sample_H = 720  # sinogram height from dataset
    cutoff = int(sample_H * 0.75)
    cond_indices = [i for i in range(sample_H) if i % 4 == 0 and i < cutoff]

    for iteration, batch in tqdm(enumerate(data_loader)):
        # batch shape: [B, T, C, H, W] = [B, 1, 1, 720, 816]
        H, W = batch.shape[-2], batch.shape[-1]

        # Build mask-based conditioning: full-resolution 2-channel image
        # Channel 0: masked sinogram (known rows filled, zeros elsewhere)
        # Channel 1: binary mask (1 = known row, 0 = unknown row)
        masked_sino = torch.zeros_like(batch)
        mask = torch.zeros_like(batch)
        masked_sino[..., cond_indices, :] = batch[..., cond_indices, :]
        mask[..., cond_indices, :] = 1.0

        # Concatenate along channel dim: [B, T, 2, H, W]
        batch_cond = torch.cat([masked_sino, mask], dim=2)

        # Target is the full sinogram: [B, T, 1, H, W]
        # Permute both to [B, C, T, H, W] for the diffusion model
        batch_label = batch.permute([0, 2, 1, 3, 4])       # [B, 1, T, H, W]
        batch_cond  = batch_cond.permute([0, 2, 1, 3, 4])  # [B, 2, T, H, W]

        result = trainer(
            batch_label,
            cond_images=batch_cond,
            unet_number=1,
            ignore_time=False,
            total_angles=H
        )
        trainer.update(unet_number=1)

        # Properly unpack tuple
        if isinstance(result, tuple):
            loss, data_loss, physics_loss = result
            data_loss_epoch.append(data_loss)
            physics_loss_epoch.append(physics_loss)
        else:
            loss = result

        loss_epoch.append(loss)

    avg_loss = sum(loss_epoch) / len(loss_epoch)
    avg_data = sum(data_loss_epoch) / len(data_loss_epoch) if data_loss_epoch else avg_loss
    avg_phys = sum(physics_loss_epoch) / len(physics_loss_epoch) if physics_loss_epoch else 0.0
    if data_loss_epoch:
        print(f"Epoch avg | data: {avg_data:.4f} "
              f"| physics: {avg_phys:.6f}")

    return trainer, (avg_loss, avg_data, avg_phys)