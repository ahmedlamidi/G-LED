import sys
import argparse
import pdb
from datetime import datetime
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import wandb
os.environ["WANDB_API_KEY"] = "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"


"""
Internal pacakage
"""
from main_seq_bfs import Args as SEQ_ARGS
from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer

sys.path.insert(0, './util')
from utils import save_args, read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
from dicom_preprocess import dicom_dataset,Efficient_LIDC_DicomDataset
sys.path.insert(0, './train_test_spatial')
from  train_diff import train_diff

class Args:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for finding the dynamics dir
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
								 default='output/mar_11_horizontal_PIML_5',
								 help='all the information of ks training')
		"""
		for diffusion model
		"""
		self.parser.add_argument("--Nt",
								 default = 1,
								 help = 'Time steps we use as a single seq')
		self.parser.add_argument("--unet_dim", 
								 default=32,
								 help='The unet dimension')
		self.parser.add_argument("--num_sample_steps", 
								 default=20,
								 help='The noise forward/reverse step')
		
		"""
		for training 
		"""
		self.parser.add_argument("--batch_size", type=int, default = 8)
		self.parser.add_argument("--epoch_num", type=int, default = 500)
		self.parser.add_argument("--device", type=str, default = "cuda:0")
		self.parser.add_argument("--shuffle",default=True)
		self.parser.add_argument("--resume", action='store_true', 
								 help='Resume training from checkpoint')
		self.parser.add_argument("--checkpoint_path", type=str, default=None,
								 help='Path to checkpoint file to resume from (default: best_model_sofar)')
		"""
		for the angular sampling pattern (cond_indices)
		"""
		self.parser.add_argument("--sample_H", type=int, default=720,
								 help='full sinogram height, i.e. number of angles over 360 deg')
		self.parser.add_argument("--angle_start", type=float, default=240,
								 help='first covered angle, in degrees')
		self.parser.add_argument("--angle_end", type=float, default=330,
								 help='end of the covered arc, in degrees (exclusive)')
		self.parser.add_argument("--angle_stride", type=int, default=1,
								 help='keep every Nth index (1 = all, 10 = every 5 degrees)')
		"""
		for the physics-informed loss
		"""
		self.parser.add_argument("--physics_loss_weight", type=float, default=0.1,
								 help='weight of the conjugate-ray symmetry loss; 0 turns it off')
		self.parser.add_argument("--physics_anchor", type=str, default='selected',
								 choices=['selected', 'global'],
								 help="'selected' anchors the symmetry on the conditioned rows, "
									  "'global' uses every row pair of the sinogram")
		"""
		for run bookkeeping
		"""
		self.parser.add_argument("--wandb_project", type=str, default='limited 90')
		self.parser.add_argument("--wandb_run_name", type=str, default=None)
		


	def update_args(self):
		args = self.parser.parse_args()
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,'diffusion_folder')
		if not os.path.isdir(args.experiment_path):
			os.makedirs(args.experiment_path)
		args.model_save_path = os.path.join(args.experiment_path,'model_save')
		if not os.path.isdir(args.model_save_path):
			os.makedirs(args.model_save_path)
		args.logging_path = os.path.join( args.experiment_path,'logging') 
		if not os.path.isdir(args.logging_path):
			os.makedirs(args.logging_path)

		args.seq_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'logging','args.txt' )
		return args

if __name__ == '__main__':

	"""
	Diff args
	"""
	diff_args = Args()
	diff_args = diff_args.update_args()
	save_args(diff_args)
	"""
	Sequence args
	"""
	seq_args = read_args_txt(SEQ_ARGS(),diff_args.seq_args_txt)
	
	"""
	Fetch dataset
	"""
	# Compute condition indices (must match main_diff_eval_bfs.py)
	sample_H = diff_args.sample_H
	angle_step_deg = 360 / sample_H  # 0.5 degrees per index at sample_H = 720
	start_idx = int(diff_args.angle_start / angle_step_deg)
	end_idx = int(diff_args.angle_end / angle_step_deg)
	cond_indices = list(range(start_idx, end_idx, diff_args.angle_stride))
	print(f'Coverage {diff_args.angle_start:g}-{diff_args.angle_end:g} deg, '
		  f'stride {diff_args.angle_stride} -> {len(cond_indices)} of {sample_H} views')
	print(f'Physics loss weight: {diff_args.physics_loss_weight}'
		  f'{"  (physics loss OFF)" if diff_args.physics_loss_weight == 0 else ""}')

	data_set = dicom_dataset(detector_count=816, angle_step=angle_step_deg,
	                         cond_indices=cond_indices)
 
	data_loader = DataLoader(dataset=data_set,
							 shuffle=diff_args.shuffle,
							 batch_size=diff_args.batch_size,
							 num_workers=2,
							 pin_memory=True)
	
	"""
	Create diffusion model
	"""
	unet1 = Unet3D(dim=diff_args.unet_dim,
				   cond_images_channels=3,  # masked sinogram + binary mask + FBP re-projection
				   memory_efficient=True,
				   dim_mults=(1, 2, 4, 8)).to(torch.device(diff_args.device))  #mid: mid channel (removed 8 to save memory)
	image_sizes = (720)   # full sinogram height (no row extraction needed)
	image_width = (816)
	imagen = ElucidatedImagen(
		unets = (unet1),
		image_sizes = image_sizes,
		image_width = image_width,   
		channels = 1,   # Match input channels (2 from sinogram duplication)     
		random_crop_sizes = None,
		num_sample_steps = diff_args.num_sample_steps, # original is 10
		cond_drop_prob = 0.1,
		sigma_min = 0.002,
		sigma_max = (80),      # max noise level, double the max noise level for upsampler  （80，160）
		sigma_data = 0.5,      # standard deviation of data distribution
		rho = 7,               # controls the sampling schedule
		P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
		P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
		S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
		S_tmin = 0.05,
		S_tmax = 50,
		S_noise = 1.003,
		condition_on_text = False,
		auto_normalize_img = False,  # Han Gao make it false
		physics_loss_weight = diff_args.physics_loss_weight,
		physics_anchor = diff_args.physics_anchor
		).to(torch.device(diff_args.device))
	trainer = ImagenTrainer(imagen, device=torch.device(diff_args.device), fp16=True)
	
	# Resume from checkpoint if specified
	if diff_args.resume:
		checkpoint_path = diff_args.checkpoint_path
		if checkpoint_path is None:
			# Default to best_model_sofar
			checkpoint_path = os.path.join(diff_args.model_save_path, 'best_model_sofar')
		if os.path.exists(checkpoint_path):
			print(f"Resuming training from checkpoint: {checkpoint_path}")
			trainer.load(checkpoint_path)
		else:
			print(f"Warning: Checkpoint not found at {checkpoint_path}, starting fresh training")
	
	# Initialize wandb
	wandb.init(
		project=diff_args.wandb_project,
		name=diff_args.wandb_run_name,
		config={
			"batch_size": diff_args.batch_size,
			"epoch_num": diff_args.epoch_num,
			"unet_dim": diff_args.unet_dim,
			"num_sample_steps": diff_args.num_sample_steps,
			"Nt": diff_args.Nt,
			"device": diff_args.device,
			"fp16": True,
			"angle_start": diff_args.angle_start,
			"angle_end": diff_args.angle_end,
			"angle_stride": diff_args.angle_stride,
			"n_cond_views": len(cond_indices),
			"physics_loss_weight": diff_args.physics_loss_weight,
			"physics_anchor": diff_args.physics_anchor,
		}
	)

	train_diff(diff_args=diff_args,
               seq_args=seq_args,
               trainer=trainer,
               data_loader=data_loader,
               cond_indices=cond_indices)
