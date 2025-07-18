"""
This file contains the basic modules for the model. 
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import transforms

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_prompt=False):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size), 
                                         requires_grad=learnable_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, first_act=False):
        super().__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction,bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(
                    *(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                                    for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                  for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, n_history=0):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                CAB(in_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
            )

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction,
                                        bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(self, x, prompt_dec, skip, history_feat: Optional[torch.Tensor] = None):
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                x = torch.cat([torch.tile(x, (1, self.n_history+1, 1, 1))], dim=1)
            else:
                x = torch.cat([x, history_feat], dim=1)

            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act,
                                                 no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)
        return x
    
    
class KspaceACSExtractor:
    '''
    Extract ACS lines from k-space data
    '''

    def __init__(self, mask_center):
        self.mask_center = mask_center
        self.low_mask_dict = {}  # avoid repeated calculation

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        get the padding size and number of low frequencies for the center mask. For fastmri and cmrxrecon dataset
        '''
        if num_low_frequencies is None or (num_low_frequencies == -1).all():
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def circular_centered_mask(self, shape, radius):
        '''
        generate a circular mask centered at the center of the image. For calgary-campinas dataset
        -shape: the shape of the mask
        -radius: the radius of the circle (ACS region)

        '''
        # radius is a tensor or int
        if type(radius) == torch.Tensor:
            # radius[0].item() # assume batch have the same radius
            radius = int(radius[0])

        center = torch.tensor(shape) // 2
        Y, X = torch.meshgrid(torch.arange(
            shape[0]), torch.arange(shape[1]), indexing='ij')
        dist_from_center = torch.sqrt(
            (X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = (dist_from_center <= radius).float()
        return mask.unsqueeze(0).unsqueeze(-1)

    def __call__(self, masked_kspace: torch.Tensor,
                 mask: torch.Tensor,
                 num_low_frequencies: Optional[int] = None,
                 mask_type: Tuple[str] = ("cartesian",),
                 ) -> torch.Tensor:
        if self.mask_center:
            mask_type = mask_type[0] # assume the same type in a batch
            mask_type = 'cartesian' if mask_type in ['uniform', 'kt_uniform', 'kt_random'] else mask_type
            if mask_type == 'kt_radial':  # cmrxrecon24 pseudo radial
                mask_low = torch.zeros_like(mask)
                b, adj_nc, h, w, two = mask.shape
                h_left = h//2 - num_low_frequencies//2
                w_left = w//2 - num_low_frequencies//2
                mask_low[:, :, h_left:h_left+num_low_frequencies, w_left:w_left+num_low_frequencies, :] \
                    = mask[:, :, h_left:h_left+num_low_frequencies, w_left:w_left+num_low_frequencies, :]
                masked_kspace_acs = masked_kspace*mask_low
            elif mask_type  == 'cartesian': # fastmri and cmrxrecon (exclude kt_radial)
                pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                    mask, num_low_frequencies
                )
                masked_kspace_acs = transforms.batched_mask_center(
                    masked_kspace, pad, pad + num_low_freqs
                )
            elif mask_type == 'poisson_disc': # cc-brain
                ss = masked_kspace.shape[-3:-1]  # (h,w)
                # * cache low mask in dict to avoid repeated calculation for the same input shape.
                if ss not in self.low_mask_dict:
                    mask_low = self.circular_centered_mask(masked_kspace.shape[-3:-1], num_low_frequencies)  # shape (1, 218, 180, 1)
                    mask_low = mask_low[None].to(masked_kspace.device)
                    self.low_mask_dict[ss] = mask_low
                else:
                    mask_low = self.low_mask_dict[ss]
                masked_kspace_acs = masked_kspace * mask_low
            else:
                raise ValueError('mask_type should be cartesian or poisson_disc')
            return masked_kspace_acs
        else:
            return masked_kspace