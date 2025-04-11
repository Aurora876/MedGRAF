from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
from pytorch_msssim import MS_SSIM
import torchvision
import sys
import copy
import argparse
import os
import glob
from os import path
import torch
import torch.optim as optim
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')

sys.path.append('submodules')        # needed to make imports work in GAN_stability

from medgraf.config import get_data, build_models, update_config, get_render_poses
from medgraf.utils import to_phi, to_theta, save_video

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config
)

from submodules.GAN_stability.gan_training import lpips
from torchvision.utils import save_image  # 确保正确导入

# 初始化感知损失
percept = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

def get_output(metrics_engine, output):
    return output[0], output[1]

def get_rays(pose, generator, img_size):
    return generator.val_ray_sampler(img_size, img_size, generator.focal, pose)[0]

def generate_image(config_file_path, xray_img_path, save_dir, model_path, 
                   img_size=128, save_every=50, psnr_stop=60, 
                   total_iterations=100, test_j=0, progress_callback=None):
    """
    封装成可调用的函数，支持从外部传入 total_iterations 和 test_j 参数。
    
    Args:
        config_file_path (str): 配置文件路径。
        xray_img_path (str): X-ray 图像路径。
        save_dir (str): 保存目录。
        model_path (str): 模型路径。
        img_size (int): 图像大小，默认 128。
        save_every (int): 保存频率，默认 50。
        psnr_stop (float): PSNR 停止阈值，默认 60。
        total_iterations (int): 总迭代次数，默认 100（从前端传入）。
        test_j (int): test 方法中的 j 值，默认 0（从前端传入）。
        progress_callback (callable): 进度回调函数。
    """
    class Args:
        pass

    args = Args()
    args.config_file = config_file_path
    args.xray_img_path = xray_img_path
    args.save_dir = save_dir
    args.model = model_path
    args.img_size = img_size
    args.save_every = save_every
    args.psnr_stop = psnr_stop

    config_file = load_config(args.config_file, '/home/zd/jzd/medgraf_vis/configs/default.yaml')

    global batch_size, out_dir, checkpoint_dir, eval_dir, device, checkpoint_io
    batch_size = 1
    out_dir = os.path.join(config_file['training']['outdir'], config_file['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, args.save_dir)
    os.makedirs(eval_dir, exist_ok=True)

    config_file['training']['nworkers'] = 0
    device = torch.device("cuda:0")
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    # 调用 reconstruct 函数，并传入 total_iterations 和 test_j
    reconstruct(args, config_file, total_iterations, test_j, progress_callback)

def reconstruct(args, config_file, total_iterations, test_j, progress_callback=None):
    """
    重构方法，支持动态设置 total_iterations 和 test_j。
    
    Args:
        args: 参数对象。
        config_file: 配置文件。
        total_iterations (int): 总迭代次数。
        test_j (int): test 方法中的 j 值。
        progress_callback (callable): 进度回调函数。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, hwfr, _ = get_data(config_file)
    config_file['data']['hwfr'] = hwfr

    # Create models
    generator, _ = build_models(config_file, disc=False)
    generator = generator.to(device)

    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    g_optim = optim.RMSprop(generator_test.parameters(), lr=0.0005, alpha=0.99, eps=1e-8)

    # Register modules to checkpoint
    checkpoint_io.register_modules(g_optimizer=g_optim)

    generator_test.eval()

    # Distributions
    ydist = get_ydist(1, device=device)  # Dummy to keep GAN training structure intact
    y = torch.zeros(batch_size).to(device)
    zdist = get_zdist(config_file['z_dist']['type'], config_file['z_dist']['dim'], device=device)

    # Load checkpoint
    model_file = args.model
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)

    psnr_engine = Engine(get_output)
    psnr = PSNR(data_range=2.)
    psnr.attach(psnr_engine, "psnr")
    ssim_engine = Engine(get_output)
    ssim = SSIM(data_range=2.)
    ssim.attach(ssim_engine, "ssim")

    N_samples = batch_size
    N_poses = 1  # corresponds to number of frames
    img_size = args.img_size

    render_radius = config_file['data']['radius']
    if isinstance(render_radius, str):  # use maximum radius
        render_radius = float(render_radius.split(',')[1])

    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    # Check if xray_img_path is a file or directory
    if os.path.isfile(args.xray_img_path):
        target_xray = [args.xray_img_path]  # Single file
    else:
        target_xray = glob.glob(os.path.join(args.xray_img_path, '*.png'))
        if not target_xray:
            raise ValueError(f"No PNG files found in directory: {args.xray_img_path}")

    print(f"Found images: {target_xray}")
    target_xray = torch.unsqueeze(trans(Image.open(target_xray[0]).convert('RGB')), 0).to(device)

    range_theta = (to_theta(config_file['data']['vmin']), to_theta(config_file['data']['vmax']))
    range_phi = (to_phi(0), to_phi(1))

    theta_mean = 0.5 * sum(range_theta)
    phi_mean = 0.5 * sum(range_phi)

    N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)  # at least 1 frame per degree

    poses = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=N_phi)

    z = zdist.sample((N_samples,))

    N_samples, N_frames = len(z), len(poses)

    z = z.unsqueeze(1).expand(-1, N_poses, -1).flatten(0, 1)

    poses = poses.unsqueeze(0) \
            .expand(N_samples, -1, -1, -1).flatten(0, 1)

    z = z.split(batch_size)

    log_rec_loss = 0.
    ssim_value = 0.
    psnr_value = 0.

    ms_ssim = MS_SSIM(data_range=2., size_average=True).to(device)

    pbar = tqdm(total=total_iterations)
    for iteration in range(total_iterations):
        g_optim.zero_grad()

        n_samples = len(z)

        rays = torch.stack([get_rays(poses[i].to(device), generator_test, img_size) for i in range(n_samples)])
        rays = rays.split(batch_size)

        rgb, depth = [], []

        for z_i, rays_i in zip(z, rays):
            bs = len(z_i)
            if rays_i is not None:
                rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)  # Bx2x(HxW)xC -> 2x(BxHxW)x3
            rgb_i, depth_i, _, _ = generator_test(z_i, rays=rays_i)

            reshape = lambda x: x.view(bs, img_size, img_size, x.shape[1]).permute(0, 3, 1, 2)  # (NxHxW)xC -> NxCxHxW
            rgb.append(reshape(rgb_i).cpu())
            depth.append(reshape(depth_i).cpu())

        rgb = torch.cat(rgb)
        depth = torch.cat(depth)

        reshape = lambda x: x.view(N_samples, N_frames, *x.shape[1:])
        xray_recons = reshape(rgb)

        # 动态调整权重
        weight_perceptual = max(0.3 - iteration / 1000, 0.1)  # 感知损失权重逐渐减少
        weight_mse = min(0.1 + iteration / 1000, 0.3)         # MSE损失权重逐渐增加

        # 调整图像尺寸以满足 ms_ssim 的要求
        target_size = (256, 256)  # 选择一个大于 160 的尺寸
        xray_recons_resized = F.interpolate(torch.unsqueeze(xray_recons[0][0], 0), size=target_size, mode='bilinear', align_corners=False)
        target_xray_resized = F.interpolate(target_xray, size=target_size, mode='bilinear', align_corners=False)

        # 计算损失
        rec_loss = weight_perceptual * percept(torch.unsqueeze(xray_recons[0][0], 0).to(device),
                                               target_xray.to(device)).sum() +\
                   weight_mse * F.mse_loss(torch.unsqueeze(xray_recons[0][0], 0).to(device), target_xray.to(device)) +\
                   0.1 * (1 - ms_ssim(xray_recons_resized.to(device), target_xray_resized.to(device)))

        rec_loss.backward()

        g_optim.step()

        log_rec_loss += rec_loss.item()

        data = torch.unsqueeze(torch.stack([
            xray_recons[0][0].unsqueeze(0).to(device),
            target_xray.to(device)
        ], 0), 0).to(device)

        psnr_state = psnr_engine.run(data)
        psnr_value += psnr_state.metrics['psnr']
        ssim_state = ssim_engine.run(data)
        ssim_value += ssim_state.metrics['ssim']

        pbar.set_description(
            f"SSIM: {ssim_value:.4f} "
            f"PSNR: {psnr_value:.4f} "
            f"Reconstruction loss g: {log_rec_loss:.4f}")
        pbar.update(1)

        # 更新进度
        if progress_callback:
            progress_callback(iteration + 1, total_iterations)

        if iteration % args.save_every == args.save_every - 1:
            test(range_phi, render_radius, theta_mean, z, generator_test, N_samples, iteration, img_size, args.save_dir, test_j)

        if psnr_value > args.psnr_stop:
            break

        ssim_value = 0.
        psnr_value = 0.
        log_rec_loss = 0.

def test(range_phi, render_radius, theta_mean, z, generator_test, N_samples, iteration, img_size, eval_dir, test_j):
    """
    测试方法，支持动态设置 test_j。
    
    Args:
        range_phi: 范围参数。
        render_radius: 渲染半径。
        theta_mean: 平均角度。
        z: 输入噪声。
        generator_test: 测试生成器。
        N_samples: 样本数量。
        iteration: 当前迭代次数。
        img_size: 图像大小。
        eval_dir: 评估目录。
        test_j (int): 动态设置的 j 值。
    """
    fps = min(int(72 / 2.), 25)  # aim for at least 2 second video
    with torch.no_grad():
        phi_rot = min(int(range_phi[1] - range_phi[0]), 72)  # at least 1 frame per degree

        poses_rot = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=phi_rot)
        zrot = z[0].clone().unsqueeze(1).expand(-1, 72, -1).flatten(0, 1)
        zrot = zrot.split(batch_size)
        samples = len(zrot)

        poses_rot = poses_rot.unsqueeze(0) \
                             .expand(samples, -1, -1, -1).flatten(0, 1)

        rays = torch.stack([get_rays(poses_rot[i].to(device), generator_test, img_size) for i in range(samples)])
        rays = rays.split(batch_size)

        rgb, depth = [], []

        for z_i, rays_i in tqdm(zip(zrot, rays), total=len(zrot), desc='Create samples...'):
            bs = len(z_i)
            if rays_i is not None:
                rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)  # Bx2x(HxW)xC -> 2x(BxHxW)x3
            rgb_i, depth_i, _, _ = generator_test(z_i, rays=rays_i)

            reshape = lambda x: x.view(bs, img_size, img_size, x.shape[1]).permute(0, 3, 1, 2)  # (NxHxW)xC -> NxCxHxW
            rgb.append(reshape(rgb_i).cpu())
            depth.append(reshape(depth_i).cpu())

        rgb = torch.cat(rgb)
        depth = torch.cat(depth)

        reshape = lambda x: x.view(N_samples, 72, *x.shape[1:])
        rgb = reshape(rgb)

        for i in range(N_samples):
            j = test_j  # 使用动态传入的 j 值
            image_path = os.path.join(eval_dir, f'sample_{i:04d}_frame_{j:03d}_rgb.png')
            save_image(rgb[i, j], image_path, normalize=True)  # 使用导入的 save_image 函数

            unique_id = '_iter_{:04d}'.format(iteration)
            modified_image_path = image_path.replace('.png', unique_id + '.png')
            os.rename(image_path, modified_image_path)