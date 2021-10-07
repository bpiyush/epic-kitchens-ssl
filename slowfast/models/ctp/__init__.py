
import torch
from torch import nn
import torch.distributed as dist

# import utils.logger
# from utils import main_utils
import yaml
import os


from slowfast.models.ctp.r3d import R3D, R2Plus1D

NUM_CLASSES = 101




# def load_checkpoint_ctp(model,pretrained):
#     # load from pre-trained, before DistributedDataParallel constructor
#     if pretrained:
#         if os.path.isfile(pretrained):
#             print("=> loading checkpoint '{}'".format(pretrained))
#             checkpoint = torch.load(pretrained, map_location="cpu")

#             # rename moco pre-trained keys
#             state_dict = checkpoint['state_dict']
#             for q,k in zip(list(model.state_dict().keys()),list(state_dict.keys())):
#                 # retain only encoder_q up to before the embedding layer
#                 #print(q,k[len("backbone."):])
#                 if k.startswith('backbone.'):# and not k.startswith('module.encoder_q.fc'):
#                     # remove prefix
#                     state_dict[k[len("backbone."):]] = state_dict[k]
#                     #state_dict[q] = state_dict[k]
#                 # delete renamed or unused k
#                 del state_dict[k]

#             #args.start_epoch = 0
#             msg = model.load_state_dict(state_dict, strict=False)
#             print(msg)
#             assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

#             print("=> loaded pre-trained model '{}'".format(pretrained))
#         else:
#             print("=> no checkpoint found at '{}'".format(pretrained))


# def build_model_ctp(feat_cfg, eval_cfg, eval_dir, args, logger):

#     # create ctp model
#     model = R2Plus1D( 
#         depth=18,
#         num_class = eval_cfg['model']['args']['n_classes'],
#         num_stages=4,
#         stem=dict(
#             temporal_kernel_size=3,
#             temporal_stride=1,
#             in_channels=3,
#             with_pool=False,
#         ),
#         down_sampling=[False, True, True, True],
#         channel_multiplier=1.0,
#         bottleneck_multiplier=1.0,
#         with_bn=True,
#         zero_init_residual=False,
#         pretrained=None,
#     )

#     # Load from checkpoint
#     checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'], feat_cfg['checkpoint'])
#     load_checkpoint_ctp(model,checkpoint_fn)

#     # Log model description
#     logger.add_line("=" * 30 + "   Model   " + "=" * 30)
#     logger.add_line(str(model))
#     logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
#     logger.add_line(main_utils.parameter_description(model))
#     logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
#     logger.add_line("File: {}".format(checkpoint_fn))

#     ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
#     return model, ckp_manager


class CTP(nn.Module):
    """
    CTP model.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(CTP, self).__init__()
        self._construct_network(cfg)
        self._init_weights()
    
    def _init_weights(self):
        ckpt_path = "/home/pbagad/models/CTP/Kinetics/pretext_checkpoint/r2p1d18_ctp_k400_epoch_90.pth"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint['state_dict']

        for q,k in zip(list(self.model.state_dict().keys()),list(state_dict.keys())):
            # retain only encoder_q up to before the embedding layer
            #print(q,k[len("backbone."):])
            if k.startswith('backbone.'):# and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
                #state_dict[q] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]


    def _construct_network(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.model = R2Plus1D(
            depth=18,
            num_class=NUM_CLASSES,
            num_stages=4,
            stem=dict(
                temporal_kernel_size=3,
                temporal_stride=1,
                in_channels=3,
                with_pool=False,
            ),
            down_sampling=[False, True, True, True],
            channel_multiplier=1.0,
            bottleneck_multiplier=1.0,
            with_bn=True,
            zero_init_residual=False,
            pretrained=None,
        )


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config

    # load cfg
    args = parse_args()
    args.cfg_file = "../../../configs/EPIC-KITCHENS/CTP_8x8_R50_k400.yaml"
    cfg = load_config(args)

    # load model
    model = CTP(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model.model(x)
    assert y.shape == (1, 101)



