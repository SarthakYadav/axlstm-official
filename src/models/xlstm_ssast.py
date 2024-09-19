import torch
import torch.nn as nn
from src.models.ssast import layernorm_wrapper
from .ssast import BaseSSAST
from .vision_lstm import SequenceTraversal, ViLBlock


__all__ = [
    "xLSTM_SSAST", "xlstm_ssast_tiny", "xlstm_ssast_small", "xlstm_ssast_medium", "xlstm_ssast_base", "xlstm_ssast_large", "xlstm_ssast_huge"
]


class xLSTM_SSAST(BaseSSAST):
    def __init__(self, 
                 img_size=(80, 200), 
                 patch_size=(16, 4), 
                 in_chans=1, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12,
                 mlp_ratio=4,
                 mask_ratio=0.5,
                 masking_mode: str = "unstructured",
                 use_cls_token=True,
                 frequency_first=True,
                 norm_layer=layernorm_wrapper,
                 alternation="bidirectional",
                 expansion_factor=2) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, mask_ratio, masking_mode, use_cls_token, frequency_first, norm_layer)

        directions = []
        if alternation == "bidirectional":
            for i in range(depth):
                if i % 2 == 0:
                    directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
                else:
                    directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        else:
            raise NotImplementedError(f"invalid alternation '{alternation}'")

        self.blocks = nn.ModuleList(
            [
                ViLBlock(
                    dim=embed_dim,
                    drop_path=0,
                    direction=directions[i],
                    expansion_factor=expansion_factor
                )
                for i in range(depth)
            ]
        )

        self.initialize_weights()


encoder_configs = {
    "tiny": {
        "depth": 12, "num_heads": 3, "embed_dim": 192
    },
    "small": {
        "depth": 12, "num_heads": 6, "embed_dim": 384
    },
    "medium": {
        "depth": 12, "num_heads": 8, "embed_dim": 512
    },
    "base": {
        "depth": 12, "num_heads": 12, "embed_dim": 768
    },
    "large": {
        "depth": 24, "num_heads": 16, "embed_dim": 1024
    },
    "huge": {
        "depth": 32, "num_heads": 16, "embed_dim": 1280
    }
}


def _get_xlstm_ssast(encoder_name, **kwargs):
    img_size = kwargs.pop("img_size", (80, 200))
    patch_size = kwargs.pop("patch_size", (16, 4))
    frequency_first = kwargs.pop("frequency_first", True)
    expansion_factor = kwargs.pop("expansion_factor", 2)
    depth_multiplier = kwargs.pop("depth_multiplier", 1)
    if expansion_factor != 2:
        print("using non-default expansion factor of:", expansion_factor)
    if depth_multiplier != 1:
        print("using non-default depth multiplier of:", depth_multiplier)

    alternation = kwargs.pop("alternation", "bidirectional")
    enc_params = encoder_configs[encoder_name]
    enc_params["depth"] *= depth_multiplier

    return xLSTM_SSAST(
        img_size=img_size,
        patch_size=patch_size,
        frequency_first=frequency_first,
        alternation=alternation,
        expansion_factor=expansion_factor,
        **enc_params, **kwargs)


def xlstm_ssast_tiny(**kwargs):
    return _get_xlstm_ssast("tiny", **kwargs)


def xlstm_ssast_small(**kwargs):
    return _get_xlstm_ssast("small", **kwargs)


def xlstm_ssast_medium(**kwargs):
    return _get_xlstm_ssast("medium", **kwargs)


def xlstm_ssast_base(**kwargs):
    return _get_xlstm_ssast("base", **kwargs)


def xlstm_ssast_large(**kwargs):
    return _get_xlstm_ssast("large", **kwargs)


def xlstm_ssast_huge(**kwargs):
    return _get_xlstm_ssast("huge", **kwargs)
