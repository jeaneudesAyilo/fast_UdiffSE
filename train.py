"""
Adapted from https://github.com/sp-uhh/sgmse/tree/main/sgmse
"""

import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl

# from pytorch_lightning.plugins import DDPPlugin
import os
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module_icp52 import SpecsDataModule

from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == "__main__":
    print("==============\nLet's start\n==============\n")

    # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument(
            "--backbone",
            type=str,
            choices=BackboneRegistry.get_all_names(),
            default="ncsnpp",
        )
        parser_.add_argument(
            "--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve"
        )
        parser_.add_argument(
            "--no_wandb",
            action="store_true",
            help="Turn off logging to W&B, using local default logger instead",
        )
        parser_.add_argument(
            "--audio_only",
            action="store_true",
            help="Specify if we won't use video modality. It is equivalent to a model using audio only",
        )

        parser_.add_argument(
            "--video_feature_type",
            type=str,
            choices=("avhubert", "raw_image", "resnet", "flow_avse"),
            default="resnet",
            help="Type of video feature representation",
        )

        parser_.add_argument(
            "--vfeat_processing_order",
            choices=[
                "default",
                "cut_extract"
            ],
            required=True,
            help="Specify in which order the video will be processed : ",
        )

        parser_.add_argument("--run_id", type=str, default="None")

        parser_.add_argument("--wandb_project", type=str, default="se-smd")

    # base_parser.add_argument("--run_id", type=str, default="None")
    # base_parser.add_argument("--wandb_project", type=str, default="se-smd")

    temp_args, _ = base_parser.parse_known_args()

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    parser = pl.Trainer.add_argparse_args(parser)
    ScoreModel.add_argparse_args(
        parser.add_argument_group("ScoreModel", description=ScoreModel.__name__)
    )
    sde_class.add_argparse_args(
        parser.add_argument_group("SDE", description=sde_class.__name__)
    )
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__)
    )
    # Add data module args
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__)
    )
    # Parse args and separate into groups
    # args, _ = parser.parse_known_args()
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    # Initialize logger, trainer, model, datamodule
    model = ScoreModel(
        backbone=args.backbone,
        sde=args.sde,
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups["ScoreModel"]),
            **vars(arg_groups["SDE"]),
            **vars(arg_groups["Backbone"]),
            **vars(arg_groups["DataModule"]),
            **{
                "audio_only": args.audio_only,
                "vfeat_processing_order": args.vfeat_processing_order,
                "video_feature_type": args.video_feature_type,
            },  ##collect in a dict all the arguments that different submodules can used in common
        },
    )

    # for u,name in enumerate(model.state_dict()):
    #     print(u,":",name)


    # Set up logger configuration
    if args.no_wandb:
        logger = TensorBoardLogger(save_dir=f"logs/{temp_args.run_id}", name="tensorboard", version =f"{temp_args.run_id}")
        #logger = TensorBoardLogger(save_dir="logs", name="tensorboard")

    else:
        if temp_args.run_id != "None":
            print(
                f" wandb.init(resume='must', id='{temp_args.run_id}', project='{temp_args.wandb_project}')"
            )

            # if os.path.isdir(f"logs/{temp_args.run_id}"):
            #     wandb.init(
            #         id=temp_args.run_id, project=temp_args.wandb_project
            #         #resume="must", id=temp_args.run_id, project=temp_args.wandb_project
            #     )

            if os.path.isdir(f"logs/{temp_args.run_id}"):
                # to continue a training
                wandb.init(
                    # id=temp_args.run_id, project=temp_args.wandb_project
                    resume="must",
                    id=temp_args.run_id,
                    project=temp_args.wandb_project,
                )
            else:
                # at the begining of training
                wandb.init(
                    id=temp_args.run_id,
                    project=temp_args.wandb_project,
                    # resume="must", id=temp_args.run_id, project=temp_args.wandb_project
                )

            print("\n Current run id: {}\n".format(wandb.run.id))

        logger = WandbLogger(project="se-smd", log_model=True, save_dir="logs")
        logger.experiment.log_code(".")

    # Set up callbacks for logger
    callbacks = [
        ModelCheckpoint(
            dirpath=f"logs/{logger.version}", save_last=True, filename="{epoch}-last"
        )
    ]


    if args.video_feature_type=="flow_avse":
        assert args.backbone == "ncsnpp_flow_avse"
        not_checking_unused_parameters = True
    else:
        not_checking_unused_parameters = False
    

    if args.backbone == "ncsnpp_flow_avse":
        assert args.video_feature_type=="flow_avse"


    # Initialize the Trainer and the DataModule
    trainer = pl.Trainer.from_argparse_args(
        arg_groups["pl.Trainer"],
        # strategy=DDPPlugin(find_unused_parameters=False),
        strategy=DDPStrategy(find_unused_parameters=not_checking_unused_parameters),
        logger=logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        accelerator="gpu",
        max_epochs=200,
    )

    # Train model
    trainer.fit(model)
