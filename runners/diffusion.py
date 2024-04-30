import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
from utils import *
# from utils.quant_util import calibrate
from utils.quant_util import QConv2d
from torch import optim
import torch.nn.functional as F
import util

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def cal_entropy(self, attn):
        return -1 * torch.sum((attn * torch.log(attn)), dim=-1).mean()

    def generate_calibrate_set(self, fpmodel, model, t_mode, num_calibrate_set):
        print("start to create calibrate set in:" + str(t_mode))
        with torch.no_grad():
            n = num_calibrate_set
            # n = self.args.timesteps
            x = torch.randn(
                n,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.device,
            )

            # x = self.sample_image(x, fpmodel)
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, self.seq, fpmodel, self.betas, eta=self.args.eta)[0]
            # print(xs[0].shape)  # 100*64*3*32*32
            if t_mode == "real":
                x = xs[-1]
            elif t_mode == "range":
                for s in range(n):
                    if s >= 100:
                        x[s] = xs[-1][s]
                    else:
                        x[s] = xs[s][s]
            elif t_mode == "random":
                shape = torch.Tensor(n)
                normal_val = torch.nn.init.normal_(shape, mean=0.4, std=0.4)*self.args.timesteps
                t = normal_val.clone().type(torch.int).to(device=self.device).clamp(0, self.args.timesteps - 1)
                # print(t)
                for s in range(n):
                    x[s] = xs[t[s]][s]
            elif t_mode == "diff":
                uncertainty = torch.zeros(self.args.timesteps).to(self.device)
                uncertainty_mark = torch.arange(0, self.args.timesteps).to(self.device)
                for k, layer in enumerate(model.modules()):
                    if type(layer) in [QConv2d]:
                        alpha = F.softmax(layer.alpha_activ, dim=1)
                        # print(alpha[0].grad)
                        _ ,group_n, dim = alpha.shape
                        for t in range(self.args.timesteps):
                            uncertainty[t] += self.cal_entropy(alpha[t]) / dim
                uncertainty -= self.args.sample_weight * self.sample_count 
                uncertainty = uncertainty[30:]
                uncertainty_mark = uncertainty_mark[30:]
                uncertainty_max = torch.max(uncertainty)
                uncertainty_max_list = uncertainty[uncertainty == uncertainty_max]
                uncertainty_mark_list = uncertainty_mark[uncertainty == uncertainty_max]
                t = uncertainty_mark_list[-1]
                self.sample_count[t] += 1 
                print(uncertainty, t)
                x = xs[t]
                self.timestep_select = t

            # x = generalized_steps_range(x, self.seq, fpmodel, self.betas, eta=self.args.eta)
            calibrate_set = inverse_data_transform(self.config, x)

            # img_id = len(glob.glob(f"{self.args.image_folder}/*"))
            # for i in range(n):
            #     tvu.save_image(
            #         calibrate_set[i], os.path.join(self.args.image_folder, f"{img_id}.png")
            #     )
            #     img_id += 1

        # print(calibrate_set.shape)  # torch.Size([batchsize, 3, 32, 32])
        return calibrate_set


    def sample(self):
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            self.seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                )
                ** 2
            )
            self.seq = [int(s) for s in list(seq)]

        model = Model(self.config, quantization=True, sequence=self.seq, args=self.args)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                if self.config.data.dataset == "CELEBA":
                    states = torch.load(
                        os.path.join(self.args.log_path, "ckpt.pth"),
                        map_location=self.config.device,
                    )
                elif self.config.data.dataset == "CIFAR10":
                    states = torch.load(
                        os.path.join(self.args.log_path, "model-790000.ckpt"),
                        map_location=self.config.device,
                    )
                elif self.config.data.dataset == "LSUN":
                    if self.config.data.category == "church_outdoor":
                        states = torch.load(
                            os.path.join(self.args.doc, "model-4432000.ckpt"),
                            map_location=self.config.device,
                        )
                    elif self.config.data.category == "bedroom":
                        states = torch.load(
                            os.path.join(self.args.doc, "model-2388000.ckpt"),
                            map_location=self.config.device,
                        )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
            if self.config.data.dataset == "CELEBA":
                states = states[-1] # ema
            state_dict = model.state_dict()
            keys = []
            for k, v in states.items():
                keys.append(k)
            i = 0
            for k, v in state_dict.items():
                if "activation_range_min" in k:
                    continue
                if "activation_range_max" in k:
                    continue
                if "x_min" in k:
                    continue
                if "x_max" in k:
                    continue
                if "groups_range" in k:
                    continue
                if "alpha_activ" in k:
                    continue
                if "mix_activ_mark1" in k:
                    continue
                # print(k, keys[i])
                if v.shape == states[keys[i]].shape:
                    state_dict[k] = states[keys[i]]
                    i = i + 1
            model.load_state_dict(state_dict, strict=False)
            # model.load_state_dict(states[0], strict=True)
            # model.load_state_dict(states, strict=False)

            # if self.config.model.ema:
            #     ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            #     ema_helper.register(model)
            #     ema_helper.load_state_dict(states[-1])
            #     ema_helper.ema(model)
            # else:
            #     ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)

        model.eval()
        
        # FP model for calibration generation
        fpmodel = Model(self.config, quantization=False)
        # for name, module in fpmodel.named_modules():
        #     print(name, module)
        fpmodel = fpmodel.to(self.device)
        # fpmodel = torch.nn.DataParallel(fpmodel, device_ids=[self.args.gpu])
        fpmodel = torch.nn.DataParallel(fpmodel)
        state_dict = fpmodel.state_dict()
        keys = []
        for k, v in states.items():
            keys.append(k)
        i = 0
        for k, v in state_dict.items():
            # print(k, keys[i])
            if v.shape == states[keys[i]].shape:
                state_dict[k] = states[keys[i]]
                i = i + 1
        fpmodel.load_state_dict(state_dict, strict=False)
            # fpmodel.load_state_dict(states, strict=False)
        fpmodel.eval()
        # generate calibrate set
        self.t_mode = self.args.calib_t_mode
        if self.config.data.dataset == "CELEBA":
            batchsize = 4
        elif self.config.data.dataset == "CIFAR10":
            batchsize = 16
        num_calibrate_set = 1024
        print("batchsize:"+str(batchsize))
        print("num_calibrate_set:"+str(num_calibrate_set))

        diff_times = int(num_calibrate_set/batchsize)
        self.sample_count = torch.zeros(self.args.timesteps).to(self.device)
        self.first_flag = True
        start_step = 0
            
        # self.first_flag = True
        if start_step == diff_times:
            start_step -= 1
            self.first_flag = True
        print(str(start_step)+"step, total"+str(diff_times))
        for step in range(start_step, diff_times):
            time_start_1 = time.time()
            calibrate_set = self.generate_calibrate_set(fpmodel, model, self.t_mode, batchsize)
            model = self.calibrate_loss(model, calibrate_set, self.device, batchsize)
            time_end_1 = time.time()
            print("running time: "+str(time_end_1 - time_start_1)+"s,"+str(start_step)+"step, total"+str(diff_times))
            self.first_flag = False

        print(self.sample_count)

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folders}/*"))
        print(f"starting from image {img_id}")
        # multi-gpu
        # node_rank = os.environ['RANK']
        # world_size = os.environ['WORLD_SIZE']
        # print(node_rank, world_size)
        # total_n_samples = 50000
        # # n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        # n_rounds = (total_n_samples - img_id) // (config.sampling.batch_size * int(world_size))
        # img_id += int(node_rank)
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folders, f"{img_id}.png")
                    )
                    img_id += 1
                    # multi-gpu
                    # img_id += int(world_size)

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folders, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folders, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
    
    def calibrate(self, model, image, device):
        print('\n==> start calibrate')
        for name, module in model.named_modules():
            if isinstance(module, QConv2d):
                module.set_calibrate(calibrate=True)
        image = image.to(device)
        print(image.shape)
        with torch.no_grad():
            self.sample_image(image, model)
        for name, module in model.named_modules():
            if isinstance(module, QConv2d):
                module.set_calibrate(calibrate=False)
        print('==> end calibrate')
        return model
    

    def calibrate_loss(self, model, image, device, batchsize):
        print('\n==> start calibrate')
        for name, module in model.named_modules():
            if isinstance(module, QConv2d):
                module.set_calibrate(calibrate=True)
                module.first_calibrate(calibrate=self.first_flag)
        image = image.to(device)

        activation_range_params = []
        for name, param in model.named_parameters():
            if "alpha_activ" in name:
                # print(name)
                param.requires_grad = True
                activation_range_params += [param]
        optimizer = torch.optim.AdamW(activation_range_params, 0.05,
                               weight_decay=0.05)
    
        from functions.denoising import generalized_steps_loss
        xs = generalized_steps_loss(image, self.seq, model, self.betas, optimizer, eta=self.args.eta
                                                , t_mode=self.t_mode, timestep_select=self.timestep_select,
                                   args=self.args)
        for name, module in model.named_modules():
            if isinstance(module, QConv2d):
                module.set_calibrate(calibrate=False)
        print('==> end calibrate')
        return model
