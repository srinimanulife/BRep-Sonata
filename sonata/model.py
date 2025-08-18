from pointcept.models import build_model
from pointcept.utils.config import Config
from huggingface_hub import hf_hub_download
import torch

def load(model_name, repo_id="Pointcept/PointTransformerV3", ckpt_name="sonata-s-scannet-2a613569.pth", **kwargs):
    """
    Loads a PointTransformerV3 model from Pointcept.
    """
    if model_name == "sonata":
        # In a real scenario, we would have a config file for the model.
        # For now, we create a minimal config to instantiate the model.
        # The config would typically be loaded from a yaml file.
        cfg = Config(dict(
            model=dict(
                type="PT-v3m2",
                in_channels=6,
                order=("z", "z-trans"),
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256, 512),
                enc_num_head=(2, 4, 8, 16, 32),
                enc_patch_size=(48, 48, 48, 48, 48),
                dec_depths=(2, 2, 2, 2),
                dec_channels=(64, 64, 128, 256),
                dec_num_head=(4, 4, 8, 16),
                dec_patch_size=(48, 48, 48, 48),
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=0.3,
                layer_scale=None,
                pre_norm=True,
                shuffle_orders=True,
                enable_rpe=False,
                enable_flash=True,
                upcast_attention=False,
                upcast_softmax=False,
                traceable=False,
                mask_token=False,
                enc_mode=False,
                freeze_encoder=False,
            )
        ))
        
        model = build_model(cfg.model)
        
        # Load pretrained weights
        try:
            weight_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
            state_dict = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {repo_id}/{ckpt_name}")
        except Exception as e:
            print(f"Could not load pretrained weights. Error: {e}")
            print("Proceeding with a randomly initialized model.")

        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
