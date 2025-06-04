# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for FEAT using PyTorch DDP.
"""
from sched import scheduler

import torch
import numpy as np

# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# import io
import os
import socket

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['MASTER_ADDR'] = 'localhost'

import math
import argparse

import torch.multiprocessing as mp
import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema,
                   requires_grad, cleanup, create_tensorboard,
                   write_tensorboard, setup_distributed, get_experiment_dir)
from tqdm import tqdm


# import models.vision_transformer as vits

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

#################################################################################
#                                  Training Loop                                #
#################################################################################


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def cleanup_ddp():
    dist.destroy_process_group()


def main(args, port):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    os.environ['MASTER_PORT'] = str(port)
    # Setup DDP:
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = rank
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}, batch_size={dist.get_world_size() * args.local_batch_size}.")

    # Setup an experiment folder:
    if rank == 0:
        if args.resume_from_checkpoint:
            experiment_dir = '/path/to/train_folder'
            checkpoint_dir = f"{experiment_dir}/checkpoints"
        else:
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/",
                                                   "-")  # e.g., EnDora-XL/2-without-dino --> EnDora-XL-2-without-dino (for naming folders)
            num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
            experiment_dir = get_experiment_dir(experiment_dir, args)
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)
    # Note that parameter initialization is done within the EnDora constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained("./vae").to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    if args.fixed_spatial:
        trainable_modules = (
            "attn_temp",
        )
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if name.endswith(tuple(trainable_modules)):
                for params in module.parameters():
                    logger.info("WARNING: Only train {} parametes!".format(name))
                    params.requires_grad = True
        logger.info("WARNING: Only train {} parametes!".format(trainable_modules))

    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    learning_rate = 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # logger.info(f"Dataset contains {len(dataset):,} videos ({args.webvideo_data_path})")
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    train_steps = 0
    first_epoch = 0
    resume_step = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if rank == 0:
            checkpoints = sorted(glob(f"{checkpoint_dir}/*.pt"),
                                 key=lambda x: int(x.split("/")[-1].split(".")[0]))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                logger.info(f"Resuming from checkpoint {checkpoint_path}")

                train_steps = int(checkpoint_path.split("/")[-1].split(".")[0])

                checkpoint = torch.load(checkpoint_path, map_location="cpu")

                model.module.load_state_dict(checkpoint["model"])
                ema.module.load_state_dict(checkpoint['ema'])

                num_update_steps_per_epoch = len(loader)
                first_epoch = train_steps // num_update_steps_per_epoch
                resume_step = train_steps % num_update_steps_per_epoch

                opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0).load_state_dict(checkpoint['opt'])
                lr_scheduler = get_scheduler(
                    name="constant",
                    optimizer=opt,
                    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
                )

                lr_scheduler.last_epoch = train_steps - 1

                logger.info(
                    f"Resuming training from step {train_steps}, epoch {first_epoch}, step_in_epoch {resume_step}")
        dist.barrier()

    for epoch in range(first_epoch, num_train_epochs):
        # print(epoch)
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)

            if args.dataset == "ucf101_img":
                image_name = video_data['image_name']
                image_names = []
                for caption in image_name:
                    single_caption = [int(item) for item in caption.split('=====')]
                    image_names.append(torch.as_tensor(single_caption))

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()

            model_kwargs = dict(use_image_num=args.use_image_num)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_mse = loss_dict["loss"].mean()
            loss = loss_mse
            loss.backward()

            if train_steps < args.start_clip_iter / dist.get_world_size():  # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                # gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                learning_rate = opt.param_groups[0]['lr']
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}/steps_per_epoch={num_update_steps_per_epoch}/max_epochs={num_train_epochs}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Learning Rate:{learning_rate:.6f}")
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                write_tensorboard(tb_writer, 'Learning Rate', learning_rate, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save EnDora checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train EnDora-XL/2 with the hyperparameters we used in our paper (except training iters).
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # num_gpus = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/col/col_train.yaml")
    if 'MASTER_PORT' not in os.environ.keys():
        port = str(find_free_network_port())
        print(f"using port {port}")
        os.environ['MASTER_PORT'] = port  # str(port)
    else:
        port = os.environ['MASTER_PORT']
        print(f"using port {port}")
    # # parser.add_argument("--config", type=str, default="./configs/cho/cho_train_with_rwkv.yaml")
    # # parser.add_argument("--port", type=int, default=6037)
    args = parser.parse_args()
    main(OmegaConf.load(args.config), port)
    # if num_gpus > 1:
    #
    #     mp.spawn(run_ddp,
    #              args=(OmegaConf.load(args.config), num_gpus),
    #              nprocs=num_gpus,
    #              join=True)
    # else:
    #     # main(OmegaConf.load(args.config), port)
    #     run_ddp(0, OmegaConf.load(args.config), num_gpus)
