from tqdm import tqdm
import torch
import sys
import os
import numpy as np
import wandb
sys.path.insert(0, './util')
from utils import save_loss

def train_diff(diff_args,
			   seq_args,
			   trainer,
			   data_loader,
			   start_epoch=0):
	loss_list = []

	loss_file = os.path.join(diff_args.logging_path, 'loss_curve.txt')

	if start_epoch > 0 and os.path.exists(loss_file):
		loss_list = list(np.loadtxt(loss_file))
		print(f"Loaded {len(loss_list)} previous loss values")

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
		model, loss = train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler)

		# Save checkpoint every 100 epochs for resumability
		if epoch > 0 and epoch % 100 == 0:
			model.save(path=os.path.join(diff_args.model_save_path,
										'checkpoint_epoch_' + str(epoch)))
			print(f"Saved checkpoint at epoch {epoch}")

		# Save latest checkpoint (overwritten each epoch for quick resume)
		model.save(path=os.path.join(diff_args.model_save_path, 'latest_checkpoint'))
		np.savetxt(os.path.join(diff_args.model_save_path, 'latest_epoch'), np.array([epoch]))

		loss_list.append(loss)
		np.savetxt(os.path.join(diff_args.logging_path, 'loss_curve.txt'), loss_list)

		# Save best model when loss improves
		if epoch >= 1 and loss < min(loss_list[:-1], default=float('inf')):
			model.save(path=os.path.join(diff_args.model_save_path,
										'best_model_sofar'))
			np.savetxt(os.path.join(diff_args.model_save_path,
								'best_model_sofar_epoch'),np.ones(2)*epoch)

		wandb.log({"epoch": epoch, "loss": loss})
		print(f"Epoch {epoch}: Loss={loss:.6f}")


def train_epoch(diff_args, seq_args, trainer, data_loader, down_sampler, up_sampler):
    loss_epoch = []

    # Pre-compute condition indices once (constant across all iterations)
    sample_H = 720  # sinogram height from dataset
    cutoff = int(sample_H * 0.75)
    cond_indices = [i for i in range(sample_H) if i % 10 == 0 and i < cutoff]

    for iteration, batch in tqdm(enumerate(data_loader)):
        # batch shape: [B, T, C, H, W] = [B, 1, 1, 720, 816]

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

        loss = trainer(
            batch_label,
            cond_images=batch_cond,
            unet_number=1,
            ignore_time=False,
        )
        trainer.update(unet_number=1)
        loss_epoch.append(loss)

    avg_loss = sum(loss_epoch) / len(loss_epoch)
    return trainer, avg_loss