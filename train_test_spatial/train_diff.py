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
	# Load previous loss list if resuming
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
		if epoch % 1 ==0 and epoch > 0:
			peep = 0
			#save_loss(diff_args, loss_list+[loss],epoch)
			#model.save(path=os.path.join(diff_args.model_save_path, 
			#							 'model_epoch_' + str(epoch)))
		
		# Save checkpoint every 10 epochs for resumability
		if epoch > 0 and epoch % 10 == 0:
			model.save(path=os.path.join(diff_args.model_save_path, 
										'checkpoint_epoch_' + str(epoch)))
			print(f"Saved checkpoint at epoch {epoch}")
		
		# Save latest checkpoint (overwritten each epoch for quick resume)
		model.save(path=os.path.join(diff_args.model_save_path, 'latest_checkpoint'))
		np.savetxt(os.path.join(diff_args.model_save_path, 'latest_epoch'), np.array([epoch]))
		
		if epoch >= 1:
			if len(loss_list) == 0 or loss < min(loss_list):
				save_loss(diff_args, loss_list+[loss],epoch)
				model.save(path=os.path.join(diff_args.model_save_path, 
											'best_model_sofar'))
				np.savetxt(os.path.join(diff_args.model_save_path, 
									'best_model_sofar_epoch'),np.ones(2)*epoch)
		loss_list.append(loss) 
		print("finish training epoch {}".format(epoch))


def train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler):
	loss_epoch = []
	print('Iteration is ', len(data_loader))
	for iteration, batch in tqdm(enumerate(data_loader)):
		#batch = batch.to(diff_args.device).float()
		bsize = batch.shape[0]
		ntime = batch.shape[1]
  
		
		# Reshape batch to [B*T, C, H, W] then back to [B, T, C, H, W]
                # batch comes as [B, T, 1, 1400, 1000] from dataset
                # batch = batch.reshape([bsize*ntime, num_velocity, 1440, 1000])
                # batch = batch.reshape([bsize, ntime, num_velocity, 1440, 1000])
		
                # Use 50% condition / 53% target split with padding to divisible by 16
                # batch shape: [B, T, C, H, W] = [B, T, 1, 720, 820]
		H, W = batch.shape[-2], batch.shape[-1]
		# Assuming [B, T, C, H, W]
		cond_width = int(W * 0.50)   # 50% condition = 410
		target_width = int(W * 0.53) # 53% target = 435
		
		batch_cond = batch[..., :cond_width]           # First 50%
		batch = batch[..., (W - target_width):]        # Last 53%
		
		# batch_cond = batch[..., :W//2]  # Left half
		# batch = batch[..., W//2:]      # Right half
		# Pad width to divisible by 16
		def pad_width_to_16(tensor):
			# tensor shape: [B, T, C, H, W]
			w = tensor.shape[-1]
			pad_w = (16 - w % 16) % 16
			if pad_w > 0:
				# Reshape to 4D for padding (F.pad reflect doesn't support 5D)
				B, T, C, H, W_orig = tensor.shape
				tensor = tensor.reshape(B * T, C, H, W_orig)
				tensor = F.pad(tensor, (0, pad_w, 0, 0), mode='reflect')
				tensor = tensor.reshape(B, T, C, H, W_orig + pad_w)
			return tensor
		
		batch_cond = pad_width_to_16(batch_cond)
		batch = pad_width_to_16(batch)            

		#need # B x F x T x H x W
		batch= batch.permute([0,2,1,3,4])
		batch_cond = batch_cond.permute([0,2,1,3,4])
		#print(batch.device)
		loss=trainer(batch,cond_images=batch_cond,unet_number=1,ignore_time=False)
		trainer.update(unet_number=1)
		loss_epoch.append(loss)
	return trainer, sum(loss_epoch)/len(loss_epoch)

