import os.path as osp
import os

import datetime
import time
import numpy as np

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from sklearn import manifold,datasets
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import copy

from scipy.stats import entropy

from sklearn.metrics.pairwise import euclidean_distances

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.ADAPT.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.ADAPT.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.ADAPT.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.ADAPT.N_CTX_TEXT
                      }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts, use_vpt = True):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, use_vpt]
        #print(combined)
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.ADAPT.N_CTX 
        dtype = clip_model.dtype  
        ctx_dim = clip_model.ln_final.weight.shape[0]   
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        domainnames_num = cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS
        domainnames = [", a {} image.".format(domain) for domain in domainnames_num]
        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(cfg.DATASET.TARGET_DOMAINS)  # number of domains
        n_dmx = cfg.TRAINER.ADAPT.N_DMX  # number of domain context
        n = n_dmx + n_ctx 
        self.n_dm = n_dm
        self.n_dmx = n_dmx
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = "a photo of a".replace("_", " ")

        if cfg.TRAINER.ADAPT.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02) 
        print("ctx vectors size: ") 
        print(ctx_vectors.shape) #[12,16,512]
        prompt_prefix = " ".join(["X"] * n) 

        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(domain_vectors, std=0.02)
        self.domain_vectors = nn.Parameter(domain_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        self.ctx = nn.Parameter(ctx_vectors) 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        naive_prompts = [
            naive_prompt_prefix + " " + name + "." for name in classnames #"a photo of a name.
        ]

        prompts = [
            prompt_prefix + " " + name + " " + domain + " an image from a domain." #"."  
            for domain in domainnames for name in classnames
        ]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        #tokenized_prompts = torch.cat([tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.ADAPT.CSC
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.naive_tokenized_prompts = naive_tokenized_prompts
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding.to(torch.device("cuda"))

    @autocast()
    def forward(self):
        ctx = self.ctx #[12,16,512]
        ctx_dim = ctx.size(-1)
        dmx = self.domain_vectors  # dm 16 512

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # dm 16 512
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1, -1)  # dm cls 16 512

        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        ctxdmx = torch.cat([ctx, dmx],
                           dim=2).reshape(self.n_cls * self.n_dm,
                                          self.n_ctx + self.n_dmx, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        neb = self.naive_embedding

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim) [24,1,512]
                ctxdmx,  # (n_cls, n_ctx, dim) [24,32,512]
                suffix,  # (n_cls, *, dim) [24,44,512]
            ],
            dim=1,
        )
        # neb = neb.to(prompts.device)
        # prompts = torch.cat([prompts, neb], dim=0)  #[24, 77, 512]

        return prompts 


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        with torch.no_grad():
            naive_embedding = self.prompt_learner.naive_embedding
            naive_embedding = naive_embedding.to("cpu")
            naive_tokenized_prompts = self.prompt_learner.naive_tokenized_prompts
            # ... 生成 prompts_naive ...
            naive_feats = self.text_encoder(naive_embedding, naive_tokenized_prompts, use_vpt=False)
            # 归一化
            naive_feats = naive_feats / naive_feats.norm(dim=-1, keepdim=True)
            
            # 【关键】注册为 buffer，这样它会自动随模型移动到 GPU，并保存到 state_dict
            self.register_buffer("naive_text_features", naive_feats)

    @autocast()
    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        with torch.no_grad():
            # 注意：这里需要确保 image_encoder 内部逻辑可以跳过 prompt 拼接
            # 或者直接使用 self.image_encoder.forward_original(image)
            image_features_original = self.image_encoder(image.type(self.dtype), use_vpt=False)

            image_features_original = image_features_original / image_features_original.norm(dim=-1, keepdim=True)

            logits_or = logit_scale * image_features_original @ self.naive_text_features.t()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))  # [32, 512]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        # return logits, image_features
        return logits, logits_or, image_features, text_features





@TRAINER_REGISTRY.register()
class ADAPT(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADAPT.PREC == "fp32" or cfg.TRAINER.ADAPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.n_dm = self.model.prompt_learner.n_dm
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder") 
        name_to_update = "prompt_learner" 
        for name, param in self.model.named_parameters():
            if name_to_update not in name: 
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        self.enabled = set() 
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.enabled.add(name)
        print(f"Parameters to be updated: {self.enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        len_train_loader_x = len(self.train_loader_x)  
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)  
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model, self.optim, self.sched) 
        self.scaler = GradScaler() if cfg.TRAINER.ADAPT.PREC == "amp" else None


        self.K=len(classnames)


        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     # print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model, device_ids=[0])

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset。
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)

        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        # self.test_batches = [int(self.num_batches * 0.33), int(self.num_batches * 0.66)]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            if self.batch_idx % 3 == 0:
                for name, param in self.model.named_parameters():
                    if "image_encoder.transformer.resblocks" and "VPT" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

            else:
                for name, param in self.model.named_parameters():
                    if "prompt_learner" in name:
                        param.requires_grad_(True)
                    elif "text_encoder.transformer.resblocks" and "VPT" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

            data_time.update(time.time() - end)

            if self.batch_idx % 3 == 0:
                loss_summary = self.forward_backward_VPT(batch_x, batch_u)
            else:
                loss_summary = self.forward_backward_prompt_learner(batch_x, batch_u)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches

                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    

    def compute_L_dom_c(self,output_x, output_u, label_x, label_u_pseudo, n_cls):
        """
        output_x: 源域 Logits [Batch, 2*K]
        output_u: 目标域 Logits [Batch, 2*K]
        label_x: 源域标签
        label_u_pseudo: 目标域伪标签
        n_cls: 类别数 K
        """
        
        def get_binary_logits(logits, labels, K):
            # 提取 p_k 和 p_{K+k}
            # pk_logits 形状 [Batch, 1]
            pk_logits = torch.gather(logits[:, :K], 1, labels.unsqueeze(1))
            pKk_logits = torch.gather(logits[:, K:2*K], 1, labels.unsqueeze(1))
            # 拼接成 [Batch, 2]，用于做二分类（源域 vs 目标域）
            return torch.cat([pk_logits, pKk_logits], dim=1)

        # 1. 提取源域和目标域的二分类 Logits
        # 源域样本在正确类别下的 [源分类器分值, 目标分类器分值]
        binary_logits_s = get_binary_logits(output_x, label_x, n_cls)
        # 目标域样本在伪标签类别下的 [源分类器分值, 目标分类器分值]
        binary_logits_u = get_binary_logits(output_u, label_u_pseudo, n_cls)

        # 2. 计算损失 (这里手动计算以实现类别平均)
        loss_c_dom = 0
        valid_counts = 0
        
        for k in range(n_cls):
            mask_s = (label_x == k)
            mask_u = (label_u_pseudo == k)
            
            if mask_s.any() and mask_u.any():
                # 源域部分: 标签应为 0 (代表源域)
                loss_s = F.cross_entropy(binary_logits_s[mask_s], 
                                        torch.zeros(mask_s.sum(), dtype=torch.long).to(output_x.device))
                # 目标域部分: 标签应为 1 (代表目标域)
                loss_u = F.cross_entropy(binary_logits_u[mask_u], 
                                        torch.ones(mask_u.sum(), dtype=torch.long).to(output_x.device))
                
                loss_c_dom += (loss_s + loss_u)
                valid_counts += 1

        return loss_c_dom / valid_counts if valid_counts > 0 else 0
    

    def compute_align_loss_soft(self, output_x, output_u, n_cls, threshold=0.8, T=2.0, mode='correlation'):
        """
        Args:
            output_x, output_u: Logits [Batch, 2*K]
            T: 温度系数 (Temperature)，T>1 会软化分布，关注相对关系
            mode: 'js' (标准JS), 'js_soft' (带温度JS), 'correlation' (皮尔逊相关)
        """
        
        def normalize_slice(logits_slice):
            # 注意：输入是 logits，不是概率
            # 在这里进行 Softmax 归一化
            return F.softmax(logits_slice, dim=1)

        def normalize_slice_with_T(logits_slice, T):
            # 带温度的 Softmax
            return F.softmax(logits_slice / T, dim=1)

        # ------------------------------------------------------
        # 1. 准备数据
        # ------------------------------------------------------
        # 切片提取 Logits (还没有做 Softmax)
        logits_x_s = output_x[:, :n_cls]
        logits_x_t = output_x[:, n_cls:]
        
        logits_u_s = output_u[:, :n_cls]
        logits_u_t = output_u[:, n_cls:]

        # 用于计算 Mask 的置信度 (始终使用原始 Logits 计算真实置信度)
        with torch.no_grad():
            p_u_s_raw = F.softmax(logits_u_s, dim=1)
            p_u_t_raw = F.softmax(logits_u_t, dim=1)
            avg_p = 0.5 * (p_u_s_raw + p_u_t_raw)
            max_conf, _ = torch.max(avg_p, dim=1)
            mask = (max_conf > threshold).float()

        # ------------------------------------------------------
        # 2. 根据模式计算 Loss
        # ------------------------------------------------------
        
        if mode == 'correlation':
            # === 方案 A: 皮尔逊相关系数 (只关注趋势) ===
            # 我们希望两个分布正相关 (Correlation -> 1)，所以 Loss = 1 - Corr
            
            def correlation_loss(x, y):
                # x, y shape: [Batch, K]
                # 减去均值
                x_mean = x - x.mean(dim=1, keepdim=True)
                y_mean = y - y.mean(dim=1, keepdim=True)
                
                # 计算分子：协方差
                numerator = (x_mean * y_mean).sum(dim=1)
                
                # 计算分母：标准差之积
                denominator = torch.sqrt((x_mean**2).sum(dim=1) * (y_mean**2).sum(dim=1)) + 1e-8
                
                # Correlation
                corr = numerator / denominator
                return 1 - corr  # Minimize (1 - Correlation)

            # 概率化 (不需要温度，直接用概率值看形状)
            p_x_s = F.softmax(logits_x_s, dim=1)
            p_x_t = F.softmax(logits_x_t, dim=1)
            p_u_s = F.softmax(logits_u_s, dim=1)
            p_u_t = F.softmax(logits_u_t, dim=1)

            loss_x = torch.mean(correlation_loss(p_x_s, p_x_t))
            
            loss_u_raw = correlation_loss(p_u_s, p_u_t)
            if mask.sum() > 0:
                loss_u = torch.sum(loss_u_raw * mask) / mask.sum()
            else:
                loss_u = torch.tensor(0.0).to(output_x.device)

        elif mode == 'js_soft':
            # === 方案 B: 带温度的 JS 散度 (软对齐) ===
            
            def compute_js(p, q):
                m = 0.5 * (p + q)
                kl_p = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=1)
                kl_q = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=1)
                return 0.5 * (kl_p + kl_q)

            # 使用带温度 T 的 Softmax
            # T 越大，分布越平滑，越只关注相对大小而非绝对值
            p_x_s = normalize_slice_with_T(logits_x_s, T)
            p_x_t = normalize_slice_with_T(logits_x_t, T)
            p_u_s = normalize_slice_with_T(logits_u_s, T)
            p_u_t = normalize_slice_with_T(logits_u_t, T)

            loss_x = torch.mean(compute_js(p_x_s, p_x_t))
            
            js_u = compute_js(p_u_s, p_u_t)
            if mask.sum() > 0:
                loss_u = torch.sum(js_u * mask) / mask.sum()
            else:
                loss_u = torch.tensor(0.0).to(output_x.device)
                
            # 因为 T 会缩小梯度，通常需要把 loss 放大 T^2 倍 (参考知识蒸馏)
            loss_x *= (T**2)
            loss_u *= (T**2)

        return loss_x + loss_u



    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy
    
    def forward_backward_prompt_learner(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.ADAPT.PREC
        if prec == "amp":
            with autocast():
                output_x, _, _, _ = self.model(image_x) #[32,36] cls=12 [source+target+pseuo]
                output_u, output_u_or, _, _ = self.model(image_u)

                output_x_p=nn.Softmax(dim=1)(output_x)
                output_u_p=nn.Softmax(dim=1)(output_u)

                #domain_x_label = torch.zeros(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))
                #domain_u_label = torch.ones(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))
                # domain_u_label = torch.zeros(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))
                # domain_x_label = torch.ones(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))

                source_domain_token_x = torch.sum(output_x_p[:, :self.n_cls], dim=1)
                #target_domain_token_x = torch.sum(output_x_p[:, self.n_cls:2 * self.n_cls], dim=1)
                #domain_token_x = torch.stack((source_domain_token_x, target_domain_token_x), dim=1)
                # domain_x_soft = torch.softmax(domain_token_x, dim=1)
                # domain_loss_x = F.cross_entropy(domain_x_soft, domain_x_label)
                #domain_loss_x = F.cross_entropy(domain_token_x, domain_x_label)
                domain_loss_x = -torch.mean(torch.log(source_domain_token_x + 1e-8))

                #source_domain_token_u = torch.sum(output_u_p[:, :self.n_cls], dim=1)
                target_domain_token_u = torch.sum(output_u_p[:, self.n_cls:2 * self.n_cls], dim=1)
                #domain_token_u = torch.stack((source_domain_token_u, target_domain_token_u), dim=1)
                # domain_u_soft = torch.softmax(domain_token_u, dim=1)
                # domain_loss_u = F.cross_entropy(domain_u_soft, domain_u_label)
                #domain_loss_u = F.cross_entropy(domain_token_u, domain_u_label)
                domain_loss_u = -torch.mean(torch.log(target_domain_token_u + 1e-8))
                
                with torch.no_grad():
                    # only clip annotation
                    pseudo_label = torch.softmax(
                        output_u_or[:, -self.n_cls:].reshape(-1, self.n_cls) /
                        self.cfg.TRAINER.ADAPT.T,
                        dim=-1)

                    max_probs, label_p = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(self.cfg.TRAINER.ADAPT.TAU).float()

                #source CE LOSS
                # output_x_soft = torch.softmax(output_x[:, :self.n_cls], dim=1)
                # loss_x = F.cross_entropy(output_x_soft, label)
                loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)

                #TARGET CE LOSS 
                # output_u_soft =torch.softmax(output_u[:, self.n_cls:2 * self.n_cls], dim=1)
                # loss_u = (F.cross_entropy(output_u_soft, label_p, reduction="none") * mask).sum() / mask.sum()
                loss_u = (F.cross_entropy(output_u[:, self.n_cls:2 * self.n_cls], label_p, reduction="none") * mask).sum() / mask.sum()

                # #IM loss
                # softmax_out = nn.Softmax(dim=1)(output_u[:, self.n_cls:2 * self.n_cls])
                # entropy_loss = torch.mean(self.Entropy(softmax_out))  
                # msoftmax = softmax_out.mean(dim=0)
                # entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-8))
                # im_loss = entropy_loss

                # with torch.no_grad():
                #     # only clip annotation
                #     pseudo_label = torch.softmax(
                #         output_u[:, self.n_cls:2 * self.n_cls].reshape(-1, self.n_cls) /
                #         self.cfg.TRAINER.ADAPT.T,
                #         dim=-1)

                #     max_probs, label_p = torch.max(pseudo_label, dim=-1)
                #     mask_l = max_probs > self.cfg.TRAINER.ADAPT.TAU # 论文中的 gamma 阈值
                #     filtered_output_u = output_u[mask_l]
                #     filtered_pseudo_labels = label_p[mask_l]

                

                loss_c_dom = self.compute_L_dom_c(output_x_p, output_u_p, label, label_p, self.n_cls)




                #lam = 2 / (1 + math.exp(-1 * 10 * self.epoch / self.max_epoch)) - 1

                loss = loss_x + self.cfg.TRAINER.ADAPT.U * loss_u + (loss_c_dom + (domain_loss_x + domain_loss_u)) 

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        # loss_summary = {
        #     "loss": loss.item(),
        #     "loss_x": loss_x.item(),
        #     "loss_u": loss_u.item(),
        #     "class_loss_u": class_loss_u.item(),
        #     "class_loss_x": class_loss_x.item(),
        #     "domain_loss_x": domain_loss_x.item(),
        #     "domain_loss_u": domain_loss_u.item(),
        #     "acc_x": compute_accuracy(output_x[:, :self.n_cls], label)[0].item(),
        # }

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "class_loss": loss_c_dom,
            "domain_loss_x": domain_loss_x.item(),
            "domain_loss_u": domain_loss_u.item(),
            "acc_x": compute_accuracy(output_x[:, :self.n_cls], label)[0].item(),
        }
        self.update_lr()

        return loss_summary
    

    def forward_backward_VPT(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.ADAPT.PREC

        if prec == "amp":
            with autocast():
                # train vision prompt
                output_x, _, _, _ = self.model(image_x)
                output_u, output_u_or, _, _ = self.model(image_u)

                output_x_p=nn.Softmax(dim=1)(output_x)
                output_u_p=nn.Softmax(dim=1)(output_u)

                #domain_x_label = torch.zeros(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))
                #domain_u_label = torch.ones(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))
                # domain_u_label = torch.zeros(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))
                # domain_x_label = torch.ones(output_x.size(0), dtype=torch.long).to(torch.device("cuda"))

                source_domain_token_x = torch.sum(output_x_p[:, :self.n_cls], dim=1)
                #target_domain_token_x = torch.sum(output_x_p[:, self.n_cls:2 * self.n_cls], dim=1)
                #domain_token_x = torch.stack((source_domain_token_x, target_domain_token_x), dim=1)
                # domain_x_soft = torch.softmax(domain_token_x, dim=1)
                # domain_loss_x = F.cross_entropy(domain_x_soft, domain_x_label)
                #domain_loss_x = F.cross_entropy(domain_token_x, domain_x_label)
                domain_loss_x = -torch.mean(torch.log(source_domain_token_x + 1e-8))

                #source_domain_token_u = torch.sum(output_u_p[:, :self.n_cls], dim=1)
                target_domain_token_u = torch.sum(output_u_p[:, self.n_cls:2 * self.n_cls], dim=1)
                #domain_token_u = torch.stack((source_domain_token_u, target_domain_token_u), dim=1)
                # domain_u_soft = torch.softmax(domain_token_u, dim=1)
                # domain_loss_u = F.cross_entropy(domain_u_soft, domain_u_label)
                #domain_loss_u = F.cross_entropy(domain_token_u, domain_u_label)
                domain_loss_u = -torch.mean(torch.log(target_domain_token_u + 1e-8))

                
                #loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)


                #IM loss
                softmax_out = nn.Softmax(dim=1)(output_u[:, self.n_cls:2 * self.n_cls])
                entropy_loss = torch.mean(self.Entropy(softmax_out))  
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-8))
                im_loss = entropy_loss

               

                # only clip annotation
                with torch.no_grad():
                    pseudo_label = torch.softmax(
                        output_u_or[:, -self.n_cls:].reshape(-1, self.n_cls) /
                        self.cfg.TRAINER.ADAPT.T,
                        dim=-1)

                    max_probs, label_p = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(self.cfg.TRAINER.ADAPT.TAU).float()

                # output_u_soft = torch.softmax(output_u[:, self.n_cls:2 * self.n_cls], dim=1)
                # loss_u = (F.cross_entropy(output_u_soft, label_p, reduction="none") * mask).sum() / mask.sum()
                loss_u = (F.cross_entropy(output_u[:, self.n_cls:2 * self.n_cls], label_p, reduction="none") * mask).sum() / mask.sum()


                # with torch.no_grad():
                #     # only clip annotation
                #     pseudo_label = torch.softmax(
                #         output_u[:, self.n_cls:2 * self.n_cls].reshape(-1, self.n_cls) /
                #         self.cfg.TRAINER.ADAPT.T,
                #         dim=-1)

                #     max_probs, label_p = torch.max(pseudo_label, dim=-1)
                #     mask_l = max_probs > self.cfg.TRAINER.ADAPT.TAU # 论文中的 gamma 阈值
                #     filtered_output_u = output_u[mask_l]
                #     filtered_pseudo_labels = label_p[mask_l]

            


                loss_c_dom = self.compute_L_dom_c(output_x_p, output_u_p, label, label_p, self.n_cls)

                #loss_align = self.compute_align_loss_soft(output_x, output_u, self.n_cls)


                #lam = 2 / (1 + math.exp(-1 * 10 * self.epoch / self.max_epoch)) - 1
                loss_G = self.cfg.TRAINER.ADAPT.U * loss_u + im_loss - (loss_c_dom + (domain_loss_x + domain_loss_u))  

            self.optim.zero_grad()
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
       

        loss_summary_G = {
            "loss_G": loss_G.item(),
            "im_loss": im_loss.item(),
            "loss_u_G": loss_u.item(),
            "domain_loss_x_G": domain_loss_x.item(),
            "domain_loss_u_G": domain_loss_u.item(),
            "class_loss_G": loss_c_dom,
            "acc_x": compute_accuracy(output_x[:, :self.n_cls], label)[0].item(),
        }
        self.update_lr()

        return loss_summary_G
            
    
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
            #     self.save_model(self.epoch,
            #                     self.output_dir,
            #                     model_name="model-best.pth.tar")
            

            self.set_model_mode("train")

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)
        if last_epoch:
            print("best result:",self.best_result,"%")

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u

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
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)[0].reshape(
                -1, self.n_dm, self.n_cls)
            # the last second slice is the logits for target domain
            output = output[:, -1, :]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
