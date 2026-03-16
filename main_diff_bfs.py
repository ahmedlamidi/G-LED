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
								 default='output/mar_11_horizontal_PIML_3',
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
		self.parser.add_argument("--batch_size", default = 8)
		self.parser.add_argument("--epoch_num", default = 500)
		self.parser.add_argument("--device", type=str, default = "cuda:0")
		self.parser.add_argument("--shuffle",default=True)
		self.parser.add_argument("--resume", action='store_true', 
								 help='Resume training from checkpoint')
		self.parser.add_argument("--checkpoint_path", type=str, default=None,
								 help='Path to checkpoint file to resume from (default: best_model_sofar)')
		


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
	data_set = dicom_dataset(detector_count=816, angle_step=(360/720))
	#1327
 
	data_loader = DataLoader(dataset=data_set,
							 shuffle=diff_args.shuffle,
							 batch_size=diff_args.batch_size,
							 num_workers=2,
							 pin_memory=True)
	
	"""
	Create diffusion model
	"""
	unet1 = Unet3D(dim=diff_args.unet_dim,
				   cond_images_channels=2,  # masked sinogram + binary mask
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
		auto_normalize_img = False  # Han Gao make it false
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
		project="G-LED-diffusion-PIML-based",
		config={
			"batch_size": diff_args.batch_size,
			"epoch_num": diff_args.epoch_num,
			"unet_dim": diff_args.unet_dim,
			"num_sample_steps": diff_args.num_sample_steps,
			"Nt": diff_args.Nt,
			"device": diff_args.device,
			"fp16": True,
		}
	)

	train_diff(diff_args=diff_args,
               seq_args=seq_args,
               trainer=trainer,
               data_loader=data_loader)
