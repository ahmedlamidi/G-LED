import os
import time

import numpy as np
import torch
import wandb


def train_diff(diff_args,
			   seq_args,
			   trainer,
			   data_loader,
			   cond_indices,
			   log,
			   record,
			   clock,
			   start_epoch=0,
			   iteration=0,
			   history=None):
	"""Logs like baselines/dolce/train.py: an `iter N: loss L (H h)` line and a
	loss.csv row every --log_every iterations, train_info.json kept up to date at
	every checkpoint, latest_checkpoint written every --ckpt_minutes. Stops early
	on the same plateau rule. history: (hours, data loss) per log line, over all
	segments."""
	history = [] if history is None else list(history)
	if cond_indices is None:
		raise ValueError("cond_indices must be provided — define it in main_diff_bfs.py and pass it through")

	device = torch.device(diff_args.device)
	save_dir, n_slices = diff_args.model_save_path, len(data_loader.dataset)

	# per-epoch curves, one value per line; a resumed run continues them
	loss_files = {k: os.path.join(diff_args.logging_path, k + '_loss.txt') for k in ('total', 'data', 'physics')}
	curves = {k: [] for k in loss_files}
	if start_epoch > 0:
		old_loss_file = os.path.join(diff_args.logging_path, 'loss_curve.txt')
		if not os.path.exists(loss_files['total']) and os.path.exists(old_loss_file):
			loss_files_read = {'total': old_loss_file}
		else:
			loss_files_read = loss_files
		for k, path in loss_files_read.items():
			if os.path.exists(path):
				curves[k] = list(np.atleast_1d(np.loadtxt(path)))[:start_epoch]
		log(f"loaded {len(curves['total'])} previous epoch losses")

	def epochs_done():
		return round(iteration * diff_args.batch_size / n_slices, 2)

	def save(name, epoch):
		"""epoch: the number of finished epochs, where a resumed run starts."""
		trainer.save(path=os.path.join(save_dir, name), epoch=epoch, iteration=iteration, hours=clock.total(),
					 history=history)

	def plateaued():
		"""Mean data loss over the last plateau_hours vs the plateau_hours before that.
		The data loss, not the total: the physics term has a different size under
		each --physics_geometry, and the runs being compared must stop by one rule."""
		w, t = diff_args.plateau_hours, clock.total()
		if not w or t < max(diff_args.min_hours, 2 * w):
			return False
		last = [l for h, l in history if t - w <= h]
		prev = [l for h, l in history if t - 2 * w <= h < t - w]
		if not last or not prev:
			return False
		gain = 1 - np.mean(last) / np.mean(prev)
		if gain < diff_args.plateau_tol:
			log(f'plateau: mean data loss {np.mean(prev):.5f} -> {np.mean(last):.5f} over the last '
				f'{2 * w:g} h ({100 * gain:+.1f}%, below {100 * diff_args.plateau_tol:g}%)')
			return True
		return False

	# Conditioning is built on the GPU from this one mask:
	# Channel 0: masked sinogram (known rows filled, zeros elsewhere)
	# Channel 1: binary mask (1 = known row, 0 = unknown row)
	# Channel 2: FBP re-projection (coarse full-sinogram estimate)
	row_mask = torch.zeros(diff_args.sample_H, 1, device=device)
	row_mask[cond_indices] = 1.0

	window, n_window, last_loss = torch.zeros(3, device=device), 0, None   # total, data, physics
	last_save = time.time()
	for epoch in range(start_epoch, diff_args.epoch_num):
		epoch_sum, n_epoch = torch.zeros(3, device=device), 0
		for batch, fbp_reproj, det_spacing in data_loader:
			# batch, fbp_reproj: [B, T, 1, H, W], det_spacing: [B]
			batch = batch.to(device, non_blocking=True)
			fbp_reproj = fbp_reproj.to(device, non_blocking=True)
			det_spacing = det_spacing.to(device, non_blocking=True)

			batch_cond = torch.cat([batch * row_mask, row_mask.expand_as(batch), fbp_reproj], dim=2)

			# Target is the full sinogram; both go to [B, C, T, H, W] for the diffusion model
			result = trainer(
				batch.permute([0, 2, 1, 3, 4]),
				cond_images=batch_cond.permute([0, 2, 1, 3, 4]),
				unet_number=1,
				ignore_time=False,
				total_angles=batch.shape[-2],
				# Anchors the conjugate-ray symmetry loss on the measured angles
				# rather than over the whole sinogram.
				cond_indices=cond_indices,
				det_spacing=det_spacing
			)
			trainer.update(unet_number=1)
			iteration += 1

			# the losses stay on the GPU until a log line needs them
			losses = torch.stack([torch.as_tensor(v, dtype=torch.float, device=device) for v in result])
			window, n_window = window + losses, n_window + 1
			epoch_sum, n_epoch = epoch_sum + losses, n_epoch + 1

			plateau = False
			if iteration % diff_args.log_every == 0:
				last_loss, data_loss, physics_loss = (window / n_window).tolist()
				history.append((clock.total(), data_loss))
				record.loss(iteration=iteration, hours=clock.total(), loss=last_loss,
							data_loss=data_loss, physics_loss=physics_loss)
				log(f'iter {iteration}: loss {last_loss:.5f} ({clock.total():.2f} h) | '
					f'data {data_loss:.5f}, physics {physics_loss:.6f}, epoch {epoch}')
				wandb.log({"iteration": iteration, "hours": clock.total(), "loss": last_loss,
						   "iter_data_loss": data_loss, "iter_physics_loss": physics_loss})
				window, n_window = torch.zeros(3, device=device), 0
				plateau = plateaued()

			finished = clock.run_over() or plateau
			if finished or clock.segment_over() or time.time() - last_save > diff_args.ckpt_minutes * 60:
				save('latest_checkpoint', epoch)
				last_save = time.time()
				record.update(clock.segment(), iterations=iteration, epochs=epochs_done(), last_loss=last_loss)
			if finished:
				reason = 'plateau' if plateau else 'hours_cap'
				record.finish(reason, clock.segment(), iterations=iteration, epochs=epochs_done(),
							  last_loss=last_loss)
				log(f'sdflow: training finished at iteration {iteration} ({reason})')
				return
			if clock.segment_over():
				log(f'sdflow: segment over at iteration {iteration}; resubmit with --resume')
				return

		total_loss, data_loss, physics_loss = (epoch_sum / max(n_epoch, 1)).tolist()
		best_before = min(curves['total'], default=float('inf'))
		for k, v in zip(('total', 'data', 'physics'), (total_loss, data_loss, physics_loss)):
			curves[k].append(v)
			np.savetxt(loss_files[k], curves[k])

		# Save best model when total loss improves
		if epoch >= 1 and total_loss < best_before:
			save('best_model_sofar', epoch + 1)

		wandb.log({
			"epoch": epoch,
			"total_loss": total_loss,
			"data_loss": data_loss,
			"physics_loss": physics_loss,
		})
		log(f'epoch {epoch} done: loss {total_loss:.5f}, data {data_loss:.5f}, physics {physics_loss:.6f}')

	save('latest_checkpoint', diff_args.epoch_num)
	record.finish('epoch_num', clock.segment(), iterations=iteration, epochs=epochs_done(), last_loss=last_loss)
	log(f'sdflow: training finished at iteration {iteration} (epoch_num)')
