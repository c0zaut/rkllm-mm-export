from data.make_input_embeds_for_quantize import MakeInputEmbeds
import numpy as np
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from rknn.api import RKNN
from rkllm.api import RKLLM
from tqdm import tqdm
import torch

quantize = MakeInputEmbeds()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    quantize.path,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(quantize.path, trust_remote_code=True, use_fast=False)

grid_t = 1
grid_h = 28
grid_w = 28
merge_size = 2
channel = 3
temporal_patch_size = 2
patch_size = 14

def export_onnx(image):
    patches = image.repeat(temporal_patch_size, 1, 1, 1)
    patches = patches.reshape(grid_t, temporal_patch_size, channel, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    model.visual.forward = forward_new(model.visual)
    feature = model.visual(flatten_patches)
    return feature

def forward_new(self):
    def tmp (hidden_states, grid_thw=None):
        hidden_states = self.patch_embed(hidden_states)
        if grid_thw is not None:
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            np.save("./rotary_pos_emb.npy", rotary_pos_emb.cpu().detach().numpy())
            np.save("./cu_seqlens.npy", cu_seqlens.cpu().detach().numpy())
        else:
            rotary_pos_emb = torch.from_numpy(np.load("./rotary_pos_emb.npy")).to(dtype=hidden_states.dtype, device=hidden_states.device)
            cu_seqlens = torch.from_numpy(np.load("./cu_seqlens.npy")).to(dtype=torch.int32, device=hidden_states.device)
        
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)
    return tmp

pixel_values = torch.randn(1, 3, 392, 392, device="cpu", dtype=torch.float32)
model.forward = export_onnx
model = model.to(torch.float32).eval()
os.makedirs("onnx", exist_ok=True)
onnx_path = f"./onnx/{quantize.model_name}.onnx"
torch.onnx.export(model, pixel_values, onnx_path, opset_version=18)

target_platform = "rk3588"

rknn = RKNN(verbose=False)
rknn.config(target_platform=target_platform, mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]], std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]])
rknn.load_onnx(onnx_path)
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
savepath = f'./rknn/{quantize.model_name}.rkllm'
rknn.export_rknn(savepath.format(target_platform))

llm = RKLLM()

ret = llm.load_huggingface(model=quantize.path, device='cpu')
if ret != 0:
    print('Load model failed!')
    exit(ret)

dataset = 'data/inputs.json'

qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8',
                quantized_algorithm='normal', target_platform='rk3588', num_npu_core=3, 
                extra_qparams=qparams, dataset=dataset)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# # Export rkllm model
ret = llm.export_rkllm(savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)
