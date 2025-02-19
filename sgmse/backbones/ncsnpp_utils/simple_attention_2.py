from typing import Optional
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from . import layerspp

conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1

import sys

sys.path.append("./sgmse/backbones/ncsnpp_utils")

sys.path.append("./sgmse/backbones/ncsnpp_utils/binary_stochastic_neurons")


from binary_stochastic_neurons.utils import Hardsigmoid
from binary_stochastic_neurons.activations import (
    DeterministicBinaryActivation,
    StochasticBinaryActivation,
)


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    Taken and adapted from https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head. For the moment, we know that don't multiple heads
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """

        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head
        self.is_inplace = is_inplace

        # Attention scaling factor
        self.scale = d_head**-0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(
            d_model, d_attn, bias=False
        )  # For the moment, we will not use multiple heads, n_heads=1 => d_attn=d_head
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, channel, d_model * width], is nothing else that the height, or the resolution`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond] ie [batch_size, nb_frame, d_cond]`
        """

        b, c, h, w = x.shape

        # Transpose from `[batch_size, channels, d_model, width]`
        # to `[batch_size, channels, widths, d_model]`
        x = x.permute(0, 1, 3, 2).contiguous()

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        attn_output = self.normal_attention(q, k, v)

        # Transpose from `[batch_size, channels, widths, d_model]`
        # to `[batch_size, channels, d_model, width]`

        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()

        return attn_output

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors of shape `[batch_size, c, w = nb_frame, h=d_attn]`
        :param k: are the keys vectors before splitting heads, of shape `[batch_size, h'= nb_frame, w' = d_attn] (cond is given)`
        :param v: are the values vectors before splitting heads, of shape `[batch_size, h'= nb_frame, w' = d_attn] (cond is given)`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.matmul(
            q, torch.unsqueeze(k, 1).transpose(-2, -1)
        )  # torch.einsum('bcij,bckj->bcik',q,torch.cat(c*[torch.unsqueeze(k,1)],1))* self.scale

        # Compute softmax
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        out = torch.matmul(
            attn, torch.unsqueeze(v, 1)
        )  # torch.einsum('bcij,bcjk->bcik',attn, torch.cat(c*[torch.unsqueeze(k,1)],1))

        # Reshape to `[batch_size, height * width, n_heads * d_head]`

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)
        


##BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond)
##SpatialTransformer(channels=in_ch, n_heads=self.n_heads, n_layers=self.tf_layers, d_cond=video_embed))

# CrossAttention(d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True)
# CrossAttention(d_model, d_model, n_heads, d_head) ##self attn
# CrossAttention(d_model, d_cond, n_heads, d_head)

# so use : CrossAttention(d_model=all_resolutions[i_level], d_cond=768/512/256 (avhubert/resnet/raw_image), n_heads=1, d_head= d_model=all_resolutions[i_level]//n_heads = in_ch )


class GaussianNoise(nn.Module):
    def __init__(self, stddev, withgrad=True):
        super().__init__()
        self.stddev = stddev
        self.withgrad = withgrad

    def forward(self, din):
        if self.training:
            # return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
            if self.withgrad:
                return din + torch.autograd.Variable(
                    torch.randn(din.size()).cuda() * self.stddev
                )
            else:
                return din + torch.randn(din.size()).cuda() * self.stddev
        return din


def span_mask(x, n_mask, l_mask, p_cond):
    """
    Apply masking to a given spectrogram.

    Parameters:
    x (torch.tensor): a tensor that stems from the spectrogram after some conv layer. shape (batchsize,channel,"frequency","time").
    n_mask (int): Number of frames to mask
    l_mask (int): Minimum length of contiguous frames to mask
    p_cond (float): Probability of masking a given frame

    Returns:
    torch.tensor: The masked tensor. Note that we may not have exactly n_mask masked time frames.bcoz some columns already masked can be re-selected 
    """
    # Copy the spectrogram to avoid modifying the original
    masked_x = torch.clone(x)
    num_time_frames = x.shape[-1]

    # Calculate the number of masking windows based on n_mask and l_mask
    num_masks = n_mask // l_mask
    for _ in range(num_masks):
        if np.random.rand() < p_cond:
            # Randomly choose a start frame for the mask
            start_frame = np.random.randint(0, num_time_frames - l_mask)
            # Apply the mask
            masked_x[:, :, :, start_frame:start_frame + l_mask] = 0  # Masking the time bins for this time span
    
    return masked_x

def mask(x, p):
    """
    Mask an exact proportion of p columns randomly in the input x of shape [b,c,h,w]
    """
    
    mask = torch.ones(x.shape).to(x)
    a = int(np.ceil(x.shape[3] * p))  # x.shape[3] time dimension of x ; or x.shape[-1]

    rt = random.sample(list(range(x.shape[3])), k=a)  ## index where we want to zero out
    mask[:, :, :, rt] = 0.0
    xm = x * mask

    return xm


class Concat_CrossAttnBlock(nn.Module):

    def __init__(
        self,
        channel,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
        masking=False,
        mask_type ="random",
        p=0.2,
        add_noise=False,
        withgrad=True,
        std=0.05,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head. For the moment, we know that don't multiple heads
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        :add_noise : whether to add noise to the audio feature when concatenating
        p: proportion of nb frame to dropout
        """

        super().__init__()
        self.add_noise = add_noise
        self.masking = masking
        if add_noise == True and masking == False:
            self.noise = GaussianNoise(std, withgrad)
        if masking == True and add_noise == False:
            self.p = p
            self.mask_type = mask_type

            if mask_type == "span_mask":
                self.p_cond = 0.9 # Probability of masking a span 
                self.l_mask = 5   # Minimum length of contiguous frames to mask #5

        if masking == True and add_noise == True:
            raise ValueError(
                f"masking = {masking} and add_noise = {add_noise} is not valid simultaneously."
            )

        self.cross_attention = CrossAttention(
            d_model, d_cond, n_heads, d_head, is_inplace
        )  ##for cross attn between h:[b,c,h,w] and v:[b,h',w']
        self.conv = conv3x3(channel + channel, channel)
        self.groupnorm = nn.GroupNorm(
            num_groups=min(channel // 4, 32), num_channels=channel, eps=1e-6
        )
        self.act = nn.SiLU()

    def forward(self, x, v):
        """
        x : the audio feature in network, x.shape : [b,c,h,w]
        v : the video feature : [b,h',w']
        """

        attn_feature = self.cross_attention(x, v)

        if self.add_noise and not self.masking:
            x = self.noise(x)

        if self.masking and not self.add_noise:
            if self.training:
                if self.mask_type == "random":
                    x = mask(x, self.p)
                elif self.mask_type == "span_mask":
                    x = span_mask(x=x, n_mask=int(np.ceil(x.shape[3] * self.p)), 
                                  l_mask=self.l_mask, p_cond=self.p_cond)

        ## if self.masking = self.noise = False, just pass

        h = self.conv(torch.cat((x, attn_feature), dim=1))
        h = self.groupnorm(h)
        h = self.act(h)

        return h


class Concat_CrossAttnBlock_Light(nn.Module):

    def __init__(
        self,
        channel,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
        masking=False,
        mask_type ="random",
        p=0.2,
        add_noise=False,
        withgrad=True,
        std=0.05,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head. For the moment, we know that don't multiple heads
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        :add_noise : whether to add noise to the audio feature when concatenating
        p: proportion of nb frame to dropout
        """

        super().__init__()
        self.add_noise = add_noise
        self.masking = masking
        if add_noise == True and masking == False:
            self.noise = GaussianNoise(std, withgrad)
        if masking == True and add_noise == False:
            self.p = p
            self.mask_type = mask_type

            if mask_type == "span_mask":
                self.p_cond = 0.9 # Probability of masking a span 
                self.l_mask = 5   # Minimum length of contiguous frames to mask #5

        if masking == True and add_noise == True:
            raise ValueError(
                f"masking = {masking} and add_noise = {add_noise} is not valid simultaneously."
            )

        self.cross_attention = CrossAttention(
            d_model, d_cond, n_heads, d_head, is_inplace
        )  ##for cross attn between h:[b,c,h,w] and v:[b,h',w']
        # self.conv = conv3x3(channel + channel, channel)
        self.groupnorm = nn.GroupNorm(
            num_groups=min(channel // 4, 32), num_channels=channel, eps=1e-6
        )
        self.act = nn.SiLU()

    def forward(self, x, v):
        """
        x : the audio feature in network, x.shape : [b,c,h,w]
        v : the video feature : [b,h',w']
        """

        attn_feature = self.cross_attention(x, v)

        if self.add_noise and not self.masking:
            x = self.noise(x)

        if self.masking and not self.add_noise:
            if self.training:
                if self.mask_type == "random":
                    x = mask(x, self.p)
                elif self.mask_type == "span_mask":
                    x = span_mask(x=x, n_mask=int(np.ceil(x.shape[3] * self.p)), 
                                  l_mask=self.l_mask, p_cond=self.p_cond)

        ## if self.masking = self.noise = False, just pass

        # h = self.conv(torch.cat((x, attn_feature), dim=1))
        attn_feature = self.groupnorm(attn_feature)
        # h = self.act(x+ attn_feature)
        h = x + attn_feature
        
        return h


class SelfAttention(nn.Module):
    """Self attention layer for `n_channels`.
    Adapted from https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
    """

    def __init__(self, n_channels):
        super().__init__()
        self.query, self.key, self.value = [
            self._lin(2 * n_channels, c)
            for c in (2 * n_channels // 8, 2 * n_channels // 8, 2 * n_channels // 8)
        ]
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.linout = self._lin(2 * n_channels // 8, 2 * n_channels)
        self.convout = nn.Conv2d(2 * n_channels, n_channels, bias=False, kernel_size=1)

    def _lin(self, n_in, n_out):
        return nn.Linear(n_in, n_out, bias=False)

    def forward(self, x, v):
        # x,v have same shape : [b,c,h,w]
        x_orig = torch.cat((x, v), dim=1)
        # Notation from the paper : arXiv:1805.08318v2
        b, c_2, h, w = x_orig.size()
        x_orig = x_orig.reshape(b, c_2, -1)  # [b,2c,h*w]
        x = x_orig.permute(0, 2, 1).contiguous()  ##[b,h*w,2c]
        q, k, v = self.query(x), self.key(x), self.value(x)  # f,g,h
        beta = F.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=-1)
        o = self.linout(torch.bmm(beta, v))  # [b,h*w,2c]
        o = self.gamma * o + x
        o = o.permute(0, 2, 1).contiguous()  # [b,2c,h*w]
        o = o.reshape(b, c_2, h, w).contiguous()  # [b,2c,h,w]
        o = self.convout(o)
        return o


class Basic_CrossAttention_Channel(nn.Module):

    def __init__(
        self,
        channel: int,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
    ):

        super().__init__()
        self.crossattention = CrossAttention(
            d_model, d_cond, n_heads, d_head, is_inplace
        )
        self.conv = conv3x3(channel, 1)

    def forward(self, x, v):
        # x,v have same shape : [b,c,h,w]
        v = self.conv(v)
        v = torch.squeeze(v, dim=1)  # get [b,w,h]
        v = v.permute(0, 2, 1).contiguous()
        o = self.crossattention(x, v)

        return o


def zero_module(module):
    """
    Zero out the parameters of a module and return it.

    Taken from https://github.com/lllyasviel/ControlNet/blob/main/ldm/modules/diffusionmodules/util.py#L177C21-L177C21
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnGating(nn.Module):

    def __init__(
        self,
        channel,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
        masking=False,
        p=0.2,
        average_gate=False,
    ):
        """
        param d_model: is the input embedding size
        param n_heads: is the number of attention heads
        param d_head: is the size of a attention head. For the moment, we won't use multiple heads
        param d_cond: is the size of the conditional embeddings
        param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        masking : whether to apply during training a mask on the input audio feature before multplying by the gate
        average_gate : whether to average the gate over the frequency dimension before computing the final convex                 combination
        """

        super().__init__()

        self.attn_block = CrossAttention(
            d_model=d_model,
            d_cond=d_cond,
            n_heads=n_heads,
            d_head=d_head,
            is_inplace=is_inplace,
        )

        self.zeros_conv = zero_module(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        )
        self.conv_layer = nn.Conv2d(
            in_channels=2 * channel, out_channels=channel, kernel_size=1
        )
        self.act = nn.Sigmoid()
        self.masking = masking
        self.p = p
        self.average_gate = average_gate

    def forward(self, x, v):
        """
        x : the audio feature in the network, x.shape : [b,c,h,w]
        v : the video feature : [b,h',w]
        """

        attn_feature = self.attn_block(x, v)
        x_out = self.conv_layer(torch.cat((x, attn_feature), dim=1))

        h = self.zeros_conv(x_out)
        gate = self.act(h)

        if self.average_gate:
            print("######## mean gating ########")
            gate = torch.mean(gate, dim=2, keepdim=True)

        if self.masking and self.training:

            x = mask(x, self.p)

        x_out = torch.mul((1 - gate), x) + torch.mul(gate, x_out)

        return x_out


############

# https://github.com/Wizaron/binary-stochastic-neurons


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D_kactiveIOnline(SparsifyBase):
    def __init__(self, height_mask=256, p=0.20):
        super(Sparsify1D_kactiveIOnline, self).__init__()

        self.height_mask = height_mask

        self.prop = 1 - p  ## proportion of frequency bins to keep

        self.k = int(
            self.height_mask * self.prop
        )  # number of top k frequency bin to consider

    def forward(self, x):

        topval = x.topk(self.k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Concat_CrossAttnBlock_LearnedMask(nn.Module):

    def __init__(
        self,
        channel,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
        p=0.2,
        std=0.05,
    ):
        """
        stft_height: nb of "frequency bin" in the stft, in fact it decrease at each reslution in the model.
        Note that this is not exactly the nb frequency bins in the raw stft. It is just the dim h of some audio feature of size [b,c,h,w], and we assimile the height h to the nb frequency bins.
        p: proportion of frequency bins to dropout
        std : standard deviation in the gaussian noise

        """

        super().__init__()

        self.cross_attention = CrossAttention(
            d_model, d_cond, n_heads, d_head, is_inplace
        )  ##for cross attn between h:[b,c,h,w] and v:[b,h',w']
        self.conv = conv3x3(channel + channel, channel)
        self.groupnorm = nn.GroupNorm(
            num_groups=min(channel // 4, 32), num_channels=channel, eps=1e-6
        )
        self.act = nn.SiLU()

        self.masklearner = NonAdaptiveLearningMask(
            stft_height=d_model, p=0.20, std_noise=0.05
        )

    def forward(self, x, v):
        """
        x : the audio feature in network, x.shape : [b,c,h,w]
        v : the video feature : [b,h',w']
        """

        attn_feature = self.cross_attention(x, v)

        masked_x, binary_mask = self.masklearner(x)

        ## if self.masking = self.noise = False, just pass

        h = self.conv(torch.cat((masked_x, attn_feature), dim=1))
        h = self.groupnorm(h)
        h = self.act(h)

        return h, binary_mask


class NonAdaptiveLearningMask(nn.Module):

    def __init__(self, stft_height=256, p=0.20, std_noise=0.05):

        super().__init__()

        self.height_mask = stft_height  # n_fft//2 +1 , and here n_fft//2=510

        self.conv_mask = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, self.height_mask),
            groups=1,
            bias=False,
        )

        self.noise = GaussianNoise(stddev=std_noise, withgrad=True)

        self.binary_act_d = DeterministicBinaryActivation(estimator="ST")

        self.slope = 1.0

        self.linear_sp = Sparsify1D_kactiveIOnline(height_mask=stft_height, p=0.20)

    def forward(self, x):

        b, c, nb_freq_bin, nb_time_frame = x.shape

        toy_input = torch.ones((x.shape[0], 1, 1, 1)).float().cuda()

        sig_output = F.sigmoid(
            (self.conv_mask(toy_input))
        )  # (batch_size,1,1, height_mask)

        # add noise and use deterministic binary activation

        sig_output = self.noise(sig_output)
        x2 = sig_output.view(sig_output.size()[0], -1)  # [batch_size,height_mask ]
        x2 = self.linear_sp(x2)
        wta_output = x2.view_as(sig_output)

        binary_mask = self.binary_act_d(
            [wta_output, self.slope]
        )  # [batch_size,1,1, self.height_mask]

        binary_mask = torch.tile(
            binary_mask[
                :,
                :,
            ],
            (nb_time_frame, 1),
        )  # [batch_size,1,nb_time_frame , height_mask=nb_freq_bin]

        masked_x_transposed = torch.mul(binary_mask, x.permute(0, 1, 3, 2).contiguous())

        masked_x = masked_x_transposed.permute(0, 1, 3, 2).contiguous()

        return masked_x, binary_mask



class NewConcat(nn.Module):

    def __init__(
        self,        
        mask_type ="random",
        p=0.2,
        nf=128
    ):
        """
        p: proportion of nb frame to dropout
        """

        super().__init__()
  
        self.mask_type = mask_type
        if mask_type=="span_mask":
            self.p_cond = 0.9 # Probability of masking a span 
            self.l_mask = 5   # Minimum length of contiguous frames to mask #5
        self.p = p
        self.conv_video = conv3x3(1, nf) 
        self.conv_concat = conv3x3(nf*2, nf)
        self.layer_norm_avfeature = nn.GroupNorm(num_groups=min(nf// 4, 32), num_channels=nf, eps=1e-6)
        self.act = nn.SiLU()

    def forward(self, h, v):
        """
        h : the audio feature in network, x.shape : [b,c,"frequency","time"] = [b,c,h,w]
        v : the video feature : [b,1,h,w]
        """
        
        fv = self.act(self.conv_video(v)) 
        if self.training :
            if self.mask_type == "random":
                h = mask(h,self.p)
            elif self.mask_type == "span_mask":
                h= span_mask(x=h, n_mask=int(np.ceil(h.shape[3] * self.p)), 
                                    l_mask=self.l_mask, p_cond=self.p_cond)
        ##for no mask set p= 0
        av_feature = torch.cat((h, fv), dim=1)
        av_feature = self.conv_concat(av_feature)             
        av_feature = self.layer_norm_avfeature(av_feature)            
        h = self.act(av_feature)
        return h