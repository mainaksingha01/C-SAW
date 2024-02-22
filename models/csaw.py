import imp
import os.path as osp
from collections import OrderedDict
import math
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torchvision.transforms as transforms
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

device_cuda = 'cuda'


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class AdaIN_trans(nn.Module):
		def __init__(self):
				super().__init__()

		def mu(self, x):
				# print(x.shape)
				# exit()
				""" Takes a (n,c,h,w) tensor as input and returns the average across
				it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
				return torch.sum(x,(1))/(x.shape[1])

		def sigma(self, x):
				""" Takes a (n,c,h,w) tensor as input and returns the standard deviation
				across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
				the permutations are required for broadcasting"""
				return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))

		def forward(self, x, y):
				""" Takes a content embeding x and a style embeding y and changes
				transforms the mean and standard deviation of the content embedding to
				that of the style. [See eq. 8 of paper] Note the permutations are
				required for broadcasting"""
				return (self.sigma(y)*((x.permute([1,0,2])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([1,0,2])



# for ViT :
class multi_scale(nn.Module):
	def __init__(self):
		super(multi_scale,self).__init__()
		self.linear = nn.ModuleList(nn.Linear(768,512) for _ in range (12))
		self.adain=AdaIN_trans()
		self.gap=nn.AdaptiveAvgPool2d((1,1))
	def forward(self,data):
		data_prompt = []
		for i in range(len(data)):
			x_mu=self.adain.mu(data[i])
			x_mu = x_mu.to(torch.float32)
			x=self.linear[i](x_mu)
			data_prompt.append(x)
		data_prompt=torch.stack(data_prompt,1) 
		return data_prompt

class projector(nn.Module):
    def __init__(self):
        super(projector,self).__init__()
        self.adain=AdaIN_trans()
    def forward(self,im_features):
        im_prompt = []
        x_mu=self.adain.mu(im_features)
        im_prompt.append(x_mu)
        im_prompt=torch.stack(im_prompt,1) 
        return im_prompt
        

class InjectionBlock(nn.Module):
    def __init__(self, vis, ctx):
        super(InjectionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(vis, vis // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis // 16, vis, bias=False),
            nn.Sigmoid()
        )

        self.linearlayer1 = nn.Sequential(nn.Linear((vis*2),vis))
        self.linearlayer2 = nn.Sequential(nn.Linear((vis*2),ctx))
        self.gap=nn.AdaptiveAvgPool2d((1,512))
        #self.gap=nn.AdaptiveAvgPool2d(1)

    def forward(self, vis):  

        vis_f = self.gap(vis)
        attn1 = self.attention(vis_f.type(torch.float))
        mulattn1 = torch.mul(attn1, vis_f)
        resattn1 = torch.cat((mulattn1, vis_f),2)
        linear1 = self.linearlayer1(resattn1)

        attn2 = self.attention(linear1.type(torch.float))
        mulattn2 = torch.mul(attn2, vis_f)
        resattn2 = torch.cat((mulattn2, vis_f),2)
        linear2 = self.linearlayer2(resattn2)
        
        output = linear2.to(torch.float16)
        return output


class TextEncoder(nn.Module):
	def __init__(self, clip_model):
		super().__init__()
		self.transformer = clip_model.transformer
		self.positional_embedding = clip_model.positional_embedding
		self.ln_final = clip_model.ln_final
		self.text_projection = clip_model.text_projection
		self.dtype = clip_model.dtype

	def forward(self, prompts, tokenized_prompts):
		x = prompts + self.positional_embedding.type(self.dtype)
		x = x.to(torch.float16)
		x = x.permute(1, 0, 2)
		x,_ = self.transformer(x)
		x = x.permute(1, 0, 2)
		x = self.ln_final(x).type(self.dtype)
		x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
		return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CSAW.N_CTX
        ctx_init = cfg.TRAINER.CSAW.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.CSAW.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        print(f'Input size of attn: "{vis_dim}"')

        self.injection = InjectionBlock(vis_dim, ctx_dim)

        self.multi = multi_scale()

        self.projector = projector()

        self.meta_net = nn.Sequential(OrderedDict([
			("linear1", nn.Linear(vis_dim, vis_dim // 16)),
			("relu", nn.ReLU(inplace=True)),
			("linear2", nn.Linear(vis_dim // 16, vis_dim))
		]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts
    
     
    def forward(self, im_features, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx   

        multi = self.multi(data)
        im_features = im_features.unsqueeze(1)
        final_features = self.projector(im_features)
        fcs = torch.cat((multi,final_features),1) 
        bias = self.injection(fcs) 
        bias = bias.to(torch.float16)
       
        alpha1 = self.meta_net(bias.type(torch.float))
        alpha2 = self.meta_net(bias.type(torch.float))
        alpha3 = self.meta_net(bias.type(torch.float))
        alpha4 = self.meta_net(bias.type(torch.float))

        alpha = torch.cat((alpha1, alpha2, alpha3, alpha4),1)
        ctx = ctx.unsqueeze(0)   

        ctx_shifted = torch.add(ctx,alpha)     
               
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts, ctx_shifted

def jigsaw(batch_image):
    perms = np.load("./perms.npy")
    resize = T.Resize(256)
    rand_crop = T.RandomCrop(255)
    grayscale = T.Grayscale(num_output_channels=3)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    target = np.random.randint(0, perms.shape[0])
    '''def color_projection(image):
        new_image = torch.zeros_like(image)
        r, g, b = image[0], image[1], image[2]
        new_image[0] = 0.8333 * r + 0.3333 * g - 0.1667 * b
        new_image[1] = 0.3333 * r + 0.3333 * g + 0.3333 * b
        new_image[2] =-0.1667 * r + 0.3333 * g + 0.8333 * b
        return new_image
    '''
    perm = perms[target]
    reshuffled_image_list = []
    for image in batch_image:
        '''if random.random() < 0.5:
            image = grayscale(image)
            image = normalize(to_tensor(image))
        else:
            image = to_tensor(image)
            if random.random() < 0.5:
              image = color_projection(image)
            image = normalize(image)'''

        k = 0
        tile_h = math.floor(image.shape[1]/3)
        tile_w = math.floor(image.shape[2]/3)
        tiles = torch.zeros((9, 3, tile_h, tile_w), dtype=image.dtype)
        for i in range(0, tile_h*3, tile_h):
            for j in range(0, tile_w*3, tile_w):
                tiles[k] = image[:, i:i+tile_h, j:j+tile_w]
                k += 1

        k = 0
        shuffled_tiles = torch.zeros((9, 3, tile_h, tile_w), dtype=image.dtype)
        for i in perm:
            shuffled_tiles[i] = tiles[k]
            k += 1
        
        row1 = torch.cat((shuffled_tiles[0],shuffled_tiles[1], shuffled_tiles[2]), -1)
        row2 = torch.cat((shuffled_tiles[3],shuffled_tiles[4], shuffled_tiles[5]), -1)
        row3 = torch.cat((shuffled_tiles[6],shuffled_tiles[7], shuffled_tiles[8]), -1)
        reshuffled_image = torch.cat((row1,row2,row3), 1)
        reshuffled_image_list.append(reshuffled_image)
    reshuffled_image_batch = torch.zeros_like(batch_image)
    reshuffled_image_h = reshuffled_image_list[0].shape[1]
    reshuffled_image_w = reshuffled_image_list[0].shape[2]
    for i in range(len(reshuffled_image_list)):
        reshuffled_image_batch[i][:,:reshuffled_image_h,:reshuffled_image_w] = reshuffled_image_list[i]
    return reshuffled_image_batch

class img_projector(nn.Module):
	def __init__(self, cfg, clip_model):
		super().__init__()
		self.lin1=nn.Linear(13*512,2*512)
		self.relu=nn.ReLU()
		self.lin2=nn.Linear(2*512,512)
	def forward(self,img):
		img = img.view(img.shape[0], -1)
		x1 = self.lin1(img)
		x2 = self.relu(x1)
		output = self.lin2(x2)
		return output

def mask_images(images):
    batch_size, channels, height, width = images.size()
    masks = torch.randint(0, 2, (batch_size, 1, height, width))
    return masks

class UpsampleNetwork(nn.Module):
    def __init__(self):
        super(UpsampleNetwork, self).__init__()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x.unsqueeze(-1).unsqueeze(-1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_proj = img_projector(cfg, clip_model)
        self.multi = multi_scale()
        self.upsample_net = UpsampleNetwork()

    def forward(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        jigsaw_image = jigsaw(image)
        jigsaw_image = jigsaw_image.to(device_cuda) 
        
        image_features, data = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_prompts, image_ctx_shifted = self.prompt_learner(image_features, data)   
        
        jigsaw_image_features, jigsaw_data = self.image_encoder(jigsaw_image.type(self.dtype))
        jigsaw_image_features = jigsaw_image_features / jigsaw_image_features.norm(dim=-1, keepdim=True)
        jigsaw_prompts, jigsaw_ctx_shifted = self.prompt_learner(jigsaw_image_features, jigsaw_data)

        jigsaw_logits = []
        for pts_i, jigpts_i, imf_i in zip(image_prompts, jigsaw_prompts, jigsaw_image_features):
            imf_i = imf_i.to(torch.float16)
            image_text_features = self.text_encoder(pts_i, tokenized_prompts)
            image_text_features = image_text_features / image_text_features.norm(dim=-1, keepdim=True)
            jigsaw_text_features = self.text_encoder(jigpts_i, tokenized_prompts)
            jigsaw_text_features = jigsaw_text_features / jigsaw_text_features.norm(dim=-1, keepdim=True)
            average_text_features = torch.mean(torch.stack([image_text_features, jigsaw_text_features]), dim=0)
            l_i = logit_scale * imf_i @ average_text_features.t()
            jigsaw_logits.append(l_i)
        jigsaw_logits = torch.stack(jigsaw_logits)

        # upsample image features for reconstruction
        self.upsample_net = self.upsample_net.half().to(device_cuda)
        output_images = self.upsample_net(image_features)
        target_size = (224, 224)
        reconstructed_images = F.interpolate(output_images, size=target_size, mode='bilinear', align_corners=False)
        original_images = image.type(self.dtype)

        return jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label

class SSLoss(torch.nn.Module):
    def __init__(self, lambda_param: float = 5e-3, gather_distributed: bool = False):
        super(SSLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss
        
        
class CsawLoss(_Loss):
    def __init__(self, T):
        super(CsawLoss, self).__init__()
        self.T = T
        self.ssl = SSLoss()

    def forward(self, jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label):
        ce_loss = F.cross_entropy(jigsaw_logits, label)       
        
        '''''
        CRP Loss
        '''      
        transprompts = jigsaw_ctx_shifted.permute(0,2,1)
        pmul = torch.matmul(jigsaw_ctx_shifted / self.T, transprompts / self.T)

        identity = torch.eye(4).to(device_cuda)
        mask = identity.repeat(4, 1, 1).to(device_cuda)
        
        crp_loss = torch.linalg.det(torch.sub(pmul / self.T , mask / self.T)).mean()

        mse = torch.nn.MSELoss()
        mse_loss = mse(original_images, reconstructed_images) 
        
        ssl_loss = self.ssl(image_features, jigsaw_image_features)

        total_loss = ce_loss + (0.6*((0.002*ssl_loss) + mse_loss)) + ((1-0.6)*0.02*(-crp_loss))
        return total_loss


@TRAINER_REGISTRY.register()
class CSAW(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CSAW.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.CSAW.PREC == "fp32" or cfg.TRAINER.CSAW.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "multi", "image_proj", "upsample_net"]
        
        for name, param in self.model.named_parameters():
            # if name_to_update not in name:
            if not any(n in name for n in name_to_update):
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f'device_cuda : {device_cuda}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CSAW.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        print(f'gpu detected : {device_count}')
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        if cfg.LOSS.NAME == "csaw":
            self.criterion = CsawLoss(T=cfg.LOSS.T)
        else:
            raise NotImplementedError

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        cfg = self.cfg
        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.CSAW.PREC
        if prec == "amp":
            with autocast():
                jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label = model(image, label)
                total_loss = self.criterion(jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label)
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label = model(image, label)
            optim.zero_grad()
            total_loss = self.criterion(jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label)
            total_loss.sum().backward()
            optim.step()

        loss_summary = {"loss": total_loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    @torch.no_grad()

    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            jigsaw_logits, image_features, jigsaw_image_features, reconstructed_images, jigsaw_ctx_shifted, original_images, label = self.model_inference(input, label)
            self.evaluator.process(jigsaw_logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]