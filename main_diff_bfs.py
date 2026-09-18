import sys
import time
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
from baselines.common.runtime import Logger, TrainClock, TrainRecord
from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer

sys.path.insert(0, './util')
from utils import save_args, read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
from dicom_preprocess import dicom_dataset,Efficient_LIDC_DicomDataset,baselines_split_dataset
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
									 help='Path to checkpoint file to resume from '
										  '(default: latest_checkpoint, else best_model_sofar)')
		self.parser.add_argument("--slices_json", type=str, default=None,
								 help='train on a split of baselines/prepare_data.py, e.g. '
									  'data/baselines_cache/limited0-45_stride10/train/slices.json: the slices every '
									  'baseline trains on (the same for every setting). Default: the DICOMs in data/Dataset')
		self.parser.add_argument("--workers", type=int,
								 default=int(os.environ.get('SLURM_CPUS_PER_TASK', 2)),
								 help='data loader workers (default: the CPUs of the SLURM job)')
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
		self.parser.add_argument("--physics_geometry", type=str, default='fan',
								 choices=['fan', 'parallel'],
								 help="'fan' pairs the ray (theta, u) with (theta + 180 deg - 2 gamma, -u), the "
									  "conjugate ray of the fan-beam scanner the sinograms come from; 'parallel' "
									  "pairs it with (theta + 180 deg, -u), exact only for a parallel beam")
		self.parser.add_argument("--physics_anchor", type=str, default='selected',
								 choices=['selected', 'global'],
								 help="'selected' anchors the symmetry on the conditioned rows, "
									  "'global' uses every row pair of the sinogram")
		"""
		for run bookkeeping
		"""
		self.parser.add_argument("--log_every", type=int, default=100,
								 help='iterations per log line / loss.csv row')
		self.parser.add_argument("--ckpt_minutes", type=float, default=30,
								 help='minutes between latest_checkpoint saves')
		self.parser.add_argument("--hours", type=float, default=0,
								 help='stop after this much training over all jobs (0 = epoch_num only)')
		self.parser.add_argument("--plateau_hours", type=float, default=3,
								 help='stop when the mean data loss of the last N hours improved on the N hours '
									  'before by less than --plateau_tol (0 = never)')
		self.parser.add_argument("--plateau_tol", type=float, default=0.05)
		self.parser.add_argument("--min_hours", type=float, default=6, help='no plateau stop before this')
		self.parser.add_argument("--segment_hours", type=float, default=23.5,
								 help='stop this job cleanly after this long, before the SLURM limit kills it')
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
	job_start = time.time()

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
		  f'{"  (physics loss OFF)" if diff_args.physics_loss_weight == 0 else f", {diff_args.physics_geometry} beam"}')

	if diff_args.slices_json:
		data_set = baselines_split_dataset(diff_args.slices_json, cond_indices,
		                                   detector_count=816, angle_step=angle_step_deg)
	else:
		data_set = dicom_dataset(detector_count=816, angle_step=angle_step_deg,
		                         cond_indices=cond_indices, return_spacing=True)
 
	data_loader = DataLoader(dataset=data_set,
							 shuffle=diff_args.shuffle,
							 batch_size=diff_args.batch_size,
							 num_workers=diff_args.workers,
							 pin_memory=True,
							 persistent_workers=diff_args.workers > 0)
	
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
		physics_anchor = diff_args.physics_anchor,
		physics_geometry = diff_args.physics_geometry
		).to(torch.device(diff_args.device))
	trainer = ImagenTrainer(imagen, device=torch.device(diff_args.device), fp16=True)
	
	log = Logger(os.path.join(diff_args.logging_path, 'log.txt'))

	# Resume from checkpoint if specified
	start_epoch, iteration, hours_before, history = 0, 0, 0.0, []
	if diff_args.resume:
		checkpoint_path = diff_args.checkpoint_path
		if checkpoint_path is None:
			# the latest checkpoint; best_model_sofar would throw away every epoch after the best one
			checkpoint_path = os.path.join(diff_args.model_save_path, 'latest_checkpoint')
			if not os.path.exists(checkpoint_path):
				checkpoint_path = os.path.join(diff_args.model_save_path, 'best_model_sofar')
		if os.path.exists(checkpoint_path):
			loaded = trainer.load(checkpoint_path)
			if 'epoch' in loaded:
				start_epoch, iteration, hours_before = loaded['epoch'], loaded['iteration'], loaded['hours']
				history = loaded.get('history', [])
			else:
				# checkpoint from before epoch / iteration / hours were stored in it
				epoch_file = os.path.join(diff_args.model_save_path,
										  'latest_epoch' if checkpoint_path.endswith('latest_checkpoint')
										  else 'best_model_sofar_epoch')
				if os.path.exists(epoch_file):
					start_epoch = int(np.atleast_1d(np.loadtxt(epoch_file))[0]) + 1
				iteration = start_epoch * (len(data_set) // diff_args.batch_size + 1)
			log(f'resumed from {checkpoint_path} at epoch {start_epoch}, iteration {iteration} ({hours_before:.2f} h)')
		else:
			log(f'Warning: Checkpoint not found at {checkpoint_path}, starting fresh training')

	n_params = sum(p.numel() for p in unet1.parameters())
	log(f'train {len(data_set)} slices, {n_params / 1e6:.1f}M parameters, {len(cond_indices)} of {sample_H} views')
	# training hours start now; the job's clock started earlier (first job: the FBP re-projection cache)
	setup_hours = (time.time() - job_start) / 3600
	clock = TrainClock(diff_args.hours or float('inf'), diff_args.segment_hours - setup_hours, hours_before)
	physics = 'off' if diff_args.physics_loss_weight == 0 else (
		f'{diff_args.physics_loss_weight} * mean over the {diff_args.physics_anchor} rows of samples with sigma < '
		f'{imagen.physics_sigma_threshold} of (x_hat(theta, u) - x_hat(theta + 180 deg'
		f'{" - 2 atan(u / 1600)" if diff_args.physics_geometry == "fan" else ""}, -u))^2')
	record = TrainRecord(diff_args.logging_path, {
		'method': 'sdflow',
		'setting': {'angle_start': diff_args.angle_start, 'angle_end': diff_args.angle_end,
					'angle_stride': diff_args.angle_stride, 'n_views': len(cond_indices),
					'rows': f'{cond_indices[0]}..{cond_indices[-1]} step {diff_args.angle_stride}'},
		'data': {'train_slices': len(data_set), 'source': diff_args.slices_json or 'data/Dataset',
				 'setup_hours': round(setup_hours, 3)},
		'model': {'parameters': n_params, 'cfg': {'unet_dim': diff_args.unet_dim, 'dim_mults': [1, 2, 4, 8],
												  'num_sample_steps': diff_args.num_sample_steps},
				  'checkpoint_used': 'best_model_sofar, EMA weights'},
		'optimizer': {'name': 'Adam', 'lr': trainer.optim0.param_groups[0]['lr'],
					  'batch_size': diff_args.batch_size, 'precision': 'fp16', 'cond_drop_prob': 0.1},
		'loss': {'name': 'EDM denoising MSE + conjugate-ray physics loss',
				 'formula': 'mean over batch and pixels of lambda(sigma) (D_theta(x + sigma n, sigma, c) - x)^2, '
							'ln sigma ~ N(-1.2, 1.2^2); physics: ' + physics,
				 'physics_loss_weight': diff_args.physics_loss_weight,
				 'physics_geometry': diff_args.physics_geometry, 'physics_anchor': diff_args.physics_anchor,
				 'logged_every_iterations': diff_args.log_every},
		'stopping': {'epoch_num': diff_args.epoch_num, 'hours_cap': diff_args.hours or None,
					 'plateau': {'on': 'data_loss', 'window_hours': diff_args.plateau_hours,
								 'tolerance': diff_args.plateau_tol,
								 'min_hours': diff_args.min_hours} if diff_args.plateau_hours else None}},
		columns=['iteration', 'hours', 'loss', 'data_loss', 'physics_loss'])
	
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
			"physics_geometry": diff_args.physics_geometry,
		}
	)

	train_diff(diff_args=diff_args,
               seq_args=seq_args,
               trainer=trainer,
               data_loader=data_loader,
               cond_indices=cond_indices,
               log=log,
               record=record,
               clock=clock,
               start_epoch=start_epoch,
               iteration=iteration,
               history=history)
