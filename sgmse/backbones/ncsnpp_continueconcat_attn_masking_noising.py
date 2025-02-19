# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

# Adapted from https://github.com/sp-uhh/sgmse/blob/main/sgmse/backbones/ncsnpp.py by adding visual aspect

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
from .ncsnpp_utils.simple_attention_2 import (
    CrossAttention,
    Concat_CrossAttnBlock,
    Concat_CrossAttnBlock_Light,
    AttnGating,
    Concat_CrossAttnBlock_LearnedMask,   
)
from .shared import BackboneRegistry

from sgmse.util.utils_video import (
    build_extractor
)

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@BackboneRegistry.register("ncsnpp_continueconcat_attn_masking_noising")
class NCSNpp_continueconcat_attn_masking_noising(nn.Module):
    """NCSN++ model, with audiovisual fusion"""

    @staticmethod
    def add_argparse_args(parser):
        # TODO: add additional arguments of constructor, if you wish to modify them.
        return parser

    def __init__(
        self,
        scale_by_sigma=True,
        nonlinearity="swish",
        nf=128,
        ch_mult=(1, 1, 2, 2, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="output_skip",
        progressive_input="input_skip",
        progressive_combine="sum",
        init_scale=0.0,
        fourier_scale=16,
        image_size=256,
        embedding_type="fourier",
        dropout=0.0,
        centered=True,
        fusion="concat_attn_masking",
        p=0.2,
        std=0.05,
        withgrad=False,
        fusion_level="",
        average_gate=False,
        masking=False,
        mask_type = "",
        no_project_video_feature=False,
        spectogram_learning=False,
        h_and_w_video = (88,88),
        **unused_kwargs,
    ):
        super().__init__()

        self.fusion = fusion
        self.fusion_level = fusion_level
        self.p = p
        self.std = std
        self.withgrad = withgrad
        self.vfeat_processing_order = unused_kwargs["vfeat_processing_order"]
        self.video_feature_type = unused_kwargs["video_feature_type"]
        self.audio_only = unused_kwargs["audio_only"]
        # self.supervised = unused_kwargs["supervised"]
        self.no_project_video_feature = no_project_video_feature

        # if self.supervised:
        #     assert embedding_type == 'none', f"In supervised case, embedding_type should be none but found {embedding_type}"
        #     assert conditional == False, f"In supervised case, conditional should be False but found {conditional}"

        print(f"### ncsnpp audio_only {self.audio_only}")

        if not self.audio_only:

            if self.vfeat_processing_order in ["cut_extract"]:
                if self.video_feature_type in ["avhubert", "resnet"]:
                    self.feature_extractor = build_extractor(
                        video_feature_type=self.video_feature_type
                    )
                elif self.video_feature_type in ["raw_image"]:
                    self.feature_extractor = None

            if self.video_feature_type == "raw_image":
                self.new_embed_dim_video = 256
                ##remember to transpose the video feature v.transpose(-2,-1) : [h_and_w_video[0]*h_and_w_video[1], num_frames] -> [num_frames, h_and_w_video[0]*h_and_w_video[1]] #h_and_w_video[0]*h_and_w_video[1]=67*67
                self.l1 = nn.Linear(h_and_w_video[0]*h_and_w_video[1], 512)  
                self.l2 = nn.Linear(512, self.new_embed_dim_video)
                ##remember to transpose again the video feature v.transpose(-2,-1) : [num_frames, self.new_embed_dim_video] -> [self.new_embed_dim_video, num_frames] ; self.new_embed_dim_video is nothing than the new embedding size for video. in cross attn, this transpose in not needed

            ##the following is important for setting d_cond for example
            if (
                self.video_feature_type in ["resnet", "avhubert"]
                and not self.no_project_video_feature
            ):
                if self.video_feature_type == "resnet":
                    video_embed = 512
                    self.new_embed_dim_video = 256
                    self.adapt_dim_layer = torch.nn.Linear(
                        video_embed, self.new_embed_dim_video
                    )  # self.c1 = torch.nn.Conv1d(video_embed, self.new_embed_dim_video, 1, stride=1)
                if self.video_feature_type == "avhubert":
                    video_embed = 768
                    self.new_embed_dim_video = 256
                    self.adapt_dim_layer = torch.nn.Linear(
                        video_embed, self.new_embed_dim_video
                    )  # self.c1 = torch.nn.Conv1d(video_embed, self.new_embed_dim_video, 1, stride=1)

            else:
                if self.video_feature_type == "resnet":
                    self.new_embed_dim_video = 512
                elif self.video_feature_type == "avhubert":
                    self.new_embed_dim_video = 768

        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            image_size // (2**i) for i in range(num_resolutions)
        ]

        self.conditional = conditional = conditional  # noise-conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["none", "fourier", "positional"]
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        self.spectogram_learning = spectogram_learning
        if self.spectogram_learning:
            num_channels = 1
        else:
            num_channels = 2  # x.real, x.imag

        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=fourier_scale
                )
            )
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        elif embedding_type == "none":  ##for example in purely supervised case
            embed_dim = None

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional and self.embedding_type in ["fourier", "positional"]:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if embedding_type != "none" and conditional == True:
            temb_dim = nf * 4
        else:  # for example in purely supervised case
            temb_dim = None

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        if not self.audio_only:
            

            if self.fusion == "concat_attn_masking":

                Concat_CrossAttn = functools.partial(
                    Concat_CrossAttnBlock,
                    add_noise=False,
                    withgrad=False,
                    masking=True,
                    p=self.p,
                    mask_type = mask_type
                )


            elif self.fusion == "concat_attn_masking_light":

                Concat_CrossAttn = functools.partial(
                    Concat_CrossAttnBlock_Light,
                    add_noise=False,
                    withgrad=False,
                    masking=True,
                    p=self.p,
                    mask_type = mask_type
                )

            elif self.fusion == "concat_attn_noising":
                Concat_CrossAttn = functools.partial(
                    Concat_CrossAttnBlock,
                    add_noise=True,
                    withgrad=self.withgrad,
                    masking=False,
                    std=self.std,
                )

            elif self.fusion == "concat_attn":
                Concat_CrossAttn = functools.partial(
                    Concat_CrossAttnBlock,
                    add_noise=False,
                    withgrad=False,
                    masking=False,
                )

            elif self.fusion == "attn_gate":

                Concat_CrossAttn = functools.partial(
                    AttnGating, masking=masking, p=self.p, average_gate=average_gate
                )

            elif self.fusion == "concat_attn_masking_learned":

                Concat_CrossAttn = functools.partial(
                    Concat_CrossAttnBlock_LearnedMask, p=self.p, std=self.std
                )

            elif self.fusion == "simple_attn":
                Concat_CrossAttn = functools.partial(CrossAttention)


            else:
                raise ValueError(f"fusion {self.fusion} unrecognized.")   

            if self.fusion in ["concat_attn_masking","concat_attn_masking_light","concat_attn_noising","concat_attn","attn_gate","concat_attn_masking_learned","simple_attn"]:
                               
                ##n_heads, d_cond useless if attn is used
                self.d_cond = (
                    self.new_embed_dim_video
                )  ##d_cond embedding for video features
                self.n_heads = 1                           
            
            
        # Downsampling block

        channels = num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        if not self.audio_only:
            if fusion_level in ["on_spectrogram"]:
                assert self.fusion == "simple_attn"
                modules.append(
                    Concat_CrossAttn(
                        d_model=256,
                        d_cond=self.d_cond,
                        n_heads=self.n_heads,
                        d_head=256 // self.n_heads,
                    )
                )

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf

        ##attn at the begining of the network
        if not self.audio_only:
            if fusion_level in ["first_layer", "encoder_only", "enc_dec"]:
                modules.append(
                    Concat_CrossAttn(
                        channel=in_ch,
                        d_model=256,
                        d_cond=self.d_cond,
                        n_heads=self.n_heads,
                        d_head=128 // self.n_heads,
                    )
                )

        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if not self.audio_only:
                    if fusion_level in ["encoder_only", "enc_dec"]:
                        modules.append(
                            Concat_CrossAttn(
                                channel=in_ch,
                                d_model=all_resolutions[i_level + 1],
                                d_cond=self.d_cond,
                                n_heads=self.n_heads,
                                d_head=all_resolutions[i_level + 1] // self.n_heads,
                            )
                        )

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]

        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))

        if not self.audio_only:
            if fusion_level in ["encoder_only", "enc_dec"]:
                modules.append(
                    Concat_CrossAttn(
                        channel=in_ch,
                        d_model=all_resolutions[-1],
                        d_cond=self.d_cond,
                        n_heads=self.n_heads,
                        d_head=all_resolutions[-1] // self.n_heads,
                    )
                )

        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0

        # Upsampling block

        for i_level in reversed(range(num_resolutions)):

            for i_block in range(
                num_res_blocks + 1
            ):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if not self.audio_only:
                if fusion_level in ["decoder_only", "enc_dec"]:
                    modules.append(
                        Concat_CrossAttn(
                            channel=in_ch,
                            d_model=all_resolutions[i_level],
                            d_cond=self.d_cond,
                            n_heads=self.n_heads,
                            d_head=all_resolutions[i_level] // self.n_heads,
                        )
                    )

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--no-centered",
            dest="centered",
            action="store_false",
            help="The data is not centered [-1, 1]",
        )
        parser.add_argument(
            "--centered",
            dest="centered",
            action="store_true",
            help="The data is centered [-1, 1]",
        )
        parser.add_argument(
            "--fusion",
            type=str,
            required=True,
            choices=[               
                "simple_attn",
                "concat_attn",
                "concat_attn_noising",
                "concat_attn_masking",
                "concat_attn_masking_light",
                "attn_gate",
                "concat_attn_masking_learned",
            ],
            help="Choose the type of audiovisual fusion",
        )

        parser.add_argument(
            "--p",
            type=float,
            default=0.2,
            help="proportion of time frame column to mask",
        )
        parser.add_argument(
            "--masking",
            action="store_true",
            help="Whether to use masking in the attn_gate fusion or not",
        )

        parser.add_argument(
            "--mask_type",
            choices=[
                "random",
                "span_mask",
            ],
            default="random",
            help="Type of mask to be used when applying concat_attn_masking",
        )

        parser.add_argument(
            "--std", type=float, default=0.05, help="standard deviation of the noise"
        )
        parser.add_argument(
            "--withgrad",
            action="store_true",
            help="Whether to use differentiable noise or not (for the noise that could be added in the masking process in audiovisual fusion)",
        )

        parser.add_argument(
            "--progressive",
            type=str,
            default="output_skip",
            choices=["output_skip", "residual", "none"],
            help="Choose the type of progressive output",
        )
        parser.add_argument(
            "--embedding_type",
            type=str,
            default="fourier",
            choices=["fourier", "positional", "none"],
            help="Choose the type of embedding for t",
        )
        parser.add_argument(
            "--conditional",
            action="store_false",
            help="whether to use also timestep t as input of the model or not",
        )
        parser.add_argument(
            "--fusion_level",
            choices=[                
                "on_spectrogram",
                "first_layer",
                "encoder_only",
                "decoder_only",
                "enc_dec",
            ],
            default="enc_dec",
            help="Level at which the fusion occurs.",
        )
        parser.add_argument(
            "--average_gate",
            action="store_true",
            help="whether to average the gate over the frequency dimension before computing the final convex combination",
        )
        parser.add_argument(
            "--no_project_video_feature",
            action="store_true",
            help="whether to project (reduce dimention) resnet|avhubert features before using them in the model. Ideally if those feat will be used in attn, it's preferable to not project. projection is done if not specified in args",
        )

        parser.set_defaults(centered=True)
        return parser

    def forward(self, x, time_cond, v):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        if self.spectogram_learning:
            x = x
        else:  # Convert real and imaginary parts of x into 2 channel dimensions
            x = torch.cat(
                (x[:, [0], :, :].real, x[:, [0], :, :].imag), dim=1
            )  # x[:,[1],:,:].real, x[:,[1],:,:].imag

        ##processing v before fusion
        if not self.audio_only:
            
            # Normalization
            v = (v / 255 - 0.421) / 0.165

            if self.vfeat_processing_order in ["cut_extract"]:

                if self.video_feature_type == "resnet":
                    # v from the dataloader: (B,1,T,H,W)
                    with torch.no_grad():
                        v = self.feature_extractor(v, lengths=None)  # (b,t,512)

                elif self.video_feature_type == "avhubert":
                    # v from the dataloader: (B,1,T,H,W)

                    with torch.no_grad():
                        v, _ = self.feature_extractor.extract_finetune(
                            source={"video": v, "audio": None},
                            padding_mask=None,
                            output_layer=None,
                        )

                if self.video_feature_type in ["resnet", "avhubert"]:
                    v = v.transpose(-2, -1).contiguous()  # (b,512, t) or #(b,768, t)


            if self.video_feature_type == "raw_image":                
                v = v.transpose(
                    -2, -1
                ).contiguous()  ##initial v shape [batch, h_and_w_video[0]*h_and_w_video[1], nb_frames] transformed to [batch, nb_frames, h_and_w_video[0]*h_and_w_video[1]]
                v = self.l1(v)
                v = self.act(self.l2(v))
                v = v.transpose(
                    -2, -1
                ).contiguous()  ##from [batch, nb_frames, new_embed_video] to [batch, new_embed_video, nb_frames]
            else:  ##we use a pretrain video feature
                if (
                    self.video_feature_type in ["resnet", "avhubert"]
                    and not self.no_project_video_feature
                ):
                    # v = self.c1(v) # initial v shape [batch, video_embed, nb_frames] transform to [batch, new_video_embed,nb_frames]. note that instead of the conv1d, one could use the nn.Linear but, one will need to transpose in a 1st time: [batch, video_emb_size(512), nb_frames] to [batch, nb_frames,video_emb_size], use the linlayer(video_emb_size, new_video_emb_size) then back to [batch, new_video_emb_size ,nb_frames, ] with transpose.
                    v = v.transpose(-2, -1).contiguous()
                    v = self.act(self.adapt_dim_layer(v))
                    v = v.transpose(
                        -2, -1
                    ).contiguous()  # where self.adapt_dim_layer = nn.Linear(video_emb,new_video_embed)

            if "attn" in self.fusion : ## only do this transpose if we use fusion that use attn. 
                v = v.transpose(
                    -2, -1
                ).contiguous()  # (b,t,512) or #(b,t,768) or #(b,t,h_and_w_video[0]*h_and_w_video[1])
            
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        elif self.embedding_type == "none":
            pass

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        if not self.audio_only:
            if self.fusion_level in ["on_spectrogram"]:
                x = modules[m_idx](x, v)
                m_idx += 1

        # Input layer: Conv2d: 2ch -> 128ch
        h = modules[m_idx](x)
        m_idx += 1

        if not self.audio_only:
            if self.fusion_level in ["first_layer", "encoder_only", "enc_dec"]:

                h = modules[m_idx](h, v)
                m_idx += 1

        hs = [h]

        # Down path in U-Net

        for i_level in range(self.num_resolutions):

            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # Attention layer (optional)
                if (
                    h.shape[-2] in self.attn_resolutions
                ):  # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if not self.audio_only:
                    if self.fusion_level in ["encoder_only", "enc_dec"]:
                        h = modules[m_idx](h, v)
                        m_idx += 1

                if self.progressive_input == "input_skip":  # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1]  # actualy equal to: h = h
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        h = modules[m_idx](h)  # Attention block
        m_idx += 1

        if not self.audio_only:
            if self.fusion_level in ["encoder_only", "enc_dec"]:
                h = modules[m_idx](h, v)
                m_idx += 1

        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block

        for i_level in reversed(range(self.num_resolutions)):

            for i_block in range(self.num_res_blocks + 1):

                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if not self.audio_only:
                if self.fusion_level in ["decoder_only", "enc_dec"]:
                    h = modules[m_idx](h, v)
                    m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.embedding_type != "none":  ##ie except purely supervised case
            if (
                self.scale_by_sigma and False
            ):  #### adding False => the following lines won't never be executed, even if self.embedding_type != "none"
                used_sigmas = used_sigmas.reshape(
                    (x.shape[0], *([1] * len(x.shape[1:])))
                )
                h = h / used_sigmas

        # Convert back to complex number
        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:, None, :, :]
        return h



@BackboneRegistry.register("ncsnpp_continueconcat_attn_masking_noising_av_28m")
class NCSNpp_continueconcat_attn_masking_noising_av_28M(NCSNpp_continueconcat_attn_masking_noising): 
    """Tiny-scale NCSN++ model. ~28M parameters"""

    def __init__(self, **kwargs):
        super().__init__(
            nf=128,
            ch_mult=(1, 2, 2, 2),
            num_res_blocks=1,
            attn_resolutions=(0,),
            **kwargs,
        )


@BackboneRegistry.register("ncsnpp_continueconcat_attn_masking_noising_av_6m")
class NCSNpp_continueconcat_attn_masking_noising_av_6M(NCSNpp_continueconcat_attn_masking_noising):
    """Tiny-scale NCSNpp_continueconcat_attn_masking_noising_av_6M++ model. ~6M parameters"""

    def __init__(self, **kwargs):
        super().__init__(
            nf=96,
            ch_mult=(1, 1, 1, 1),
            num_res_blocks=1,
            attn_resolutions=(0,),
            **kwargs,
        )

