import argparse
import pdb
import os
import sys
import json
from torch.utils.data import DataLoader
import torch

from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from main_seq_bfs import Args as Args_seq
from main_diff_bfs import Args as Args_diff
sys.path.insert(0, './util')
from utils import read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel import SequentialModel as transformer
sys.path.insert(0, './train_test_spatial')
from test_diff import test_final_overall 
from test_diff_ensamble import test_final_overall_ensamble
from dicom_preprocess import dicom_dataset



def load_config(config_file='configurations/Limited45Sparse10.json'):
	"""
	Load configuration from JSON file with optional command-line overrides
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default=config_file,
					   help="Path to config JSON file")
	parser.add_argument("--bfs_dynamic_folder", type=str, default=None)
	parser.add_argument("--batch_size", type=int, default=None)
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--sample_H", type=int, default=None)
	parser.add_argument("--angle_start", type=int, default=None)
	parser.add_argument("--angle_end", type=int, default=None)
	parser.add_argument("--angle_stride", type=int, default=None)
	parser.add_argument("--detector_count", type=int, default=None)
	
	cli_args = parser.parse_args()
	
	# Load from JSON
	with open(cli_args.config, 'r') as f:
		config = json.load(f)
	
	# Override with command-line args if provided
	for key in config:
		cli_value = getattr(cli_args, key, None)
		if cli_value is not None:
			config[key] = cli_value
	
	# Convert to object for easier access
	class Config:
		def __init__(self, **kwargs):
			self.__dict__.update(kwargs)
	
	args = Config(**config)
	
	# Compute derived paths
	args.seq_args_txt  = os.path.join(args.bfs_dynamic_folder,
									 'logging','args.txt')
	args.diff_args_txt = os.path.join(args.bfs_dynamic_folder,
									 'diffusion_folder',
									 'logging','args.txt')
	
	# output dataset
	data_folder_name = os.path.basename(os.path.normpath(args.data_path)).replace('_', '').lower()
	args.experiment_path = os.path.join(args.bfs_dynamic_folder,
										'diffusion_folder',
									data_folder_name)
	if not os.path.isdir(args.experiment_path):
		os.makedirs(args.experiment_path)
	
	return args

if __name__ == '__main__':
	"""
	Load configuration from JSON
	"""
	args_final = load_config()
	# args_seq  = read_args_txt(Args_seq(), 
	# 						  args_final.seq_args_txt)
	# args_diff = read_args_txt(Args_diff(), 
	# 						  args_final.diff_args_txt)
	
	"""
	Fetch dataset
	"""
	# Compute condition indices (must match train_diff.py)
	angle_step_deg = 360 / args_final.sample_H  # degrees per index
	start_idx = int(args_final.angle_start / angle_step_deg)
	end_idx = int(args_final.angle_end / angle_step_deg)
	cond_indices = list(range(start_idx, end_idx, args_final.angle_stride))
	data_set = dicom_dataset(data_path=args_final.data_path, 
							 detector_count=args_final.detector_count, 
							 angle_step=(360/args_final.sample_H),
							 cond_indices=cond_indices)
	
	data_loader = DataLoader(dataset=data_set, 
							 shuffle=False,
							 batch_size=args_final.batch_size)

	
	"""
	Fetch models
	"""
	# model = transformer(args_seq).to(args_final.device).float()
	# print('Number of parameters: {}'.format(model._num_parameters()))
	# if args_final.use_best:
	# 	model.load_state_dict(torch.load(args_seq.current_model_save_path+'best_model_sofar'))
	# else:
	# 	model.load_state_dict(torch.load(args_seq.current_model_save_path+'model_epoch_'+str(args_final.Nt_read),map_location=torch.device(args_final.device)))	
	
	unet1 = Unet3D(dim=32,
				   cond_images_channels=3,  # masked sinogram + binary mask + FBP re-projection
				   memory_efficient=True,
				   dim_mults=(1, 2,4,8)).to(torch.device(args_final.device))
	image_sizes = (args_final.sample_H)
	image_width = (args_final.detector_count)
	imagen = ElucidatedImagen(
            unets = (unet1),
            image_sizes = image_sizes,
            image_width = image_width,   
            channels = 1,   # Han Gao add the input to this args explicity     
            random_crop_sizes = None,
            num_sample_steps = 20, # original is 10
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
            ).to(torch.device(args_final.device))
	trainer = ImagenTrainer(imagen, device =torch.device(args_final.device), fp16=True)
	trainer.load(path=args_final.bfs_dynamic_folder+'/best_model_sofar')
	test_final_overall(args_final,
					   None,
					   None,
					   trainer,
					   None,
					   data_loader,
					   cond_indices=cond_indices)









