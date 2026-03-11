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

    # Pre-compute indices once (constant across all iterations)
    sample_H = 720  # sinogram height from dataset
    cutoff = int(sample_H * 0.75)
    batch_cond_indices = [i for i in range(sample_H) if i % 4 == 0 and i < cutoff]
    label_indices = [i for i in range(sample_H) if not (i % 4 == 0 and i < cutoff)]
    original_pred_h = len(label_indices)
    original_cond_h = len(batch_cond_indices)

    # Pre-compute padding amount
    pred_pad = (16 - original_pred_h % 16) % 16
    cond_pad = (16 - original_cond_h % 16) % 16

    for iteration, batch in tqdm(enumerate(data_loader)):
        H, W = batch.shape[-2], batch.shape[-1]

        batch_cond = batch[..., batch_cond_indices, :]
        batch_label = batch[..., label_indices, :]

        if pred_pad > 0:
            B, T, C, Hp, Wp = batch_label.shape
            batch_label = batch_label.reshape(B * T, C, Hp, Wp)
            batch_label = F.pad(batch_label, (0, 0, 0, pred_pad), mode='replicate')
            batch_label = batch_label.reshape(B, T, C, Hp + pred_pad, Wp)
        if cond_pad > 0:
            B, T, C, Hc, Wc = batch_cond.shape
            batch_cond = batch_cond.reshape(B * T, C, Hc, Wc)
            batch_cond = F.pad(batch_cond, (0, 0, 0, cond_pad), mode='replicate')
            batch_cond = batch_cond.reshape(B, T, C, Hc + cond_pad, Wc)

        batch_label = batch_label.permute([0, 2, 1, 3, 4])
        batch_cond  = batch_cond.permute([0, 2, 1, 3, 4])

        result = trainer(
            batch_label,
            cond_images=batch_cond,
            unet_number=1,
            ignore_time=False,
            cond_indices=batch_cond_indices,
            label_indices=label_indices,
            original_pred_h=original_pred_h,
            original_cond_h=original_cond_h,
            total_angles=H
        )

        # Properly unpack tuple
        if isinstance(result, tuple):
            loss, data_loss, physics_loss = result
            data_loss_epoch.append(data_loss)
            physics_loss_epoch.append(physics_loss)
        else:
            loss = result

        loss_epoch.append(loss)

        # Log every 50 iterations
        if iteration % 50 == 0:
            d = data_loss if isinstance(result, tuple) else loss
            p = physics_loss if isinstance(result, tuple) else 0.0
            ratio = p / (d + 1e-8)
            print(f"  iter {iteration} | data: {d:.4f} | physics: {p:.6f} | ratio: {ratio:.4f}")

    avg_loss = sum(loss_epoch) / len(loss_epoch)
    avg_data = sum(data_loss_epoch) / len(data_loss_epoch) if data_loss_epoch else avg_loss
    avg_phys = sum(physics_loss_epoch) / len(physics_loss_epoch) if physics_loss_epoch else 0.0
    if data_loss_epoch:
        print(f"Epoch avg | data: {avg_data:.4f} "
              f"| physics: {avg_phys:.6f}")

    return trainer, (avg_loss, avg_data, avg_phys)