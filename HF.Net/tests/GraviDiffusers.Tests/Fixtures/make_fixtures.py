"""Tiny Stable Diffusion ONNX fixtures with a VAE factor of 8, an inpainting UNet and a LoRA.

Writes to <out>/:
  sd8/            4-channel UNet, factor-8 VAE, CLIP text encoder (ONNX, diffusers layout)
  sd8-inpaint/    same, with a 9-channel UNet
  lora-peft.safetensors   diffusers/PEFT key layout (unet. / text_encoder. prefixes)
  lora-kohya.safetensors  the same adapter in kohya's layout, alpha = rank / 2 and B doubled
  ref-*.png       reference images from diffusers' ONNX pipelines (the LoRA one from a fused export)
"""
import json, os, shutil, sys
import numpy as np
import torch
from diffusers import (AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerDiscreteScheduler, OnnxRuntimeModel,
                       OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline)
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

out = sys.argv[1]
torch.manual_seed(0)
torch_repo = snapshot_download("hf-internal-testing/tiny-stable-diffusion-torch")
onnx_repo = snapshot_download("optimum-internal-testing/tiny-stable-diffusion-onnx")

text = CLIPTextModel.from_pretrained(torch_repo + "/text_encoder").eval()
vae = AutoencoderKL(block_out_channels=[8, 8, 8, 8], down_block_types=["DownEncoderBlock2D"] * 4,
                    up_block_types=["UpDecoderBlock2D"] * 4, latent_channels=4, layers_per_block=1,
                    norm_num_groups=8, sample_size=64).eval()


def unet(in_channels):
    torch.manual_seed(1)
    return UNet2DConditionModel(block_out_channels=(16, 32), layers_per_block=1, sample_size=8, norm_num_groups=8,
                                in_channels=in_channels, out_channels=4, cross_attention_dim=32,
                                attention_head_dim=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D")).eval()


class Encoder(torch.nn.Module):
    def __init__(s, v): super().__init__(); s.vae = v
    # Samples inside the graph (RandomNormalLike), as diffusers' own exports do.
    def forward(s, sample):
        d = s.vae.encode(sample).latent_dist
        return d.mean + d.std * torch.randn_like(d.mean)


class Decoder(torch.nn.Module):
    def __init__(s, v): super().__init__(); s.vae = v
    def forward(s, latent_sample): return s.vae.decode(latent_sample).sample


def export(module, args, names, outputs, path, dynamic):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The TorchScript exporter keeps module paths in node names, which is what LoRA matching reads.
    torch.onnx.export(module, args, path, input_names=names, output_names=outputs, dynamic_axes=dynamic,
                      opset_version=17, do_constant_folding=True, dynamo=False)


def write(root, unet_model, text_model, in_channels):
    if os.path.exists(root): shutil.rmtree(root)
    shutil.copytree(onnx_repo + "/tokenizer", root + "/tokenizer")
    shutil.copytree(onnx_repo + "/scheduler", root + "/scheduler")
    ids = torch.zeros(1, 77, dtype=torch.int32)
    export(text_model, (ids,), ["input_ids"], ["last_hidden_state", "pooler_output"], root + "/text_encoder/model.onnx",
           {"input_ids": {0: "b"}})
    export(Encoder(vae), (torch.randn(1, 3, 64, 64),), ["sample"], ["latent_sample"], root + "/vae_encoder/model.onnx",
           {"sample": {0: "b", 2: "h", 3: "w"}})
    export(Decoder(vae), (torch.randn(1, 4, 8, 8),), ["latent_sample"], ["sample"], root + "/vae_decoder/model.onnx",
           {"latent_sample": {0: "b", 2: "h", 3: "w"}})
    export(unet_model, (torch.randn(2, in_channels, 8, 8), torch.tensor([10], dtype=torch.int64), torch.randn(2, 77, 32)),
           ["sample", "timestep", "encoder_hidden_states"], ["out_sample"], root + "/unet/model.onnx",
           {"sample": {0: "b", 2: "h", 3: "w"}, "encoder_hidden_states": {0: "b"}})
    vae_config = dict(vae.config); vae_config["scaling_factor"] = 0.18215
    for part in ("vae_encoder", "vae_decoder"):
        json.dump({k: v for k, v in vae_config.items() if not k.startswith("_")}, open(f"{root}/{part}/config.json", "w"), default=list)
    json.dump({"in_channels": in_channels, "out_channels": 4, "cross_attention_dim": 32}, open(root + "/unet/config.json", "w"))


class UnetRoot(torch.nn.Module):
    def __init__(s, u):
        super().__init__()
        for name, child in u.named_children(): s.add_module(name, child)
        s._u = [u]
    def forward(s, sample, timestep, encoder_hidden_states):
        return s._u[0](sample, timestep, encoder_hidden_states, return_dict=False)[0]


class TextRoot(torch.nn.Module):
    """Registers the text model's parts under text_model, as transformers 4 did and real exports read."""
    def __init__(s, t):
        super().__init__()
        inner = torch.nn.Module()
        for name, child in t.named_children(): inner.add_module(name, child)
        s.add_module("text_model", inner)
        s._t = [t]
    def forward(s, input_ids):
        o = s._t[0](input_ids.long())
        return o.last_hidden_state, o.pooler_output


base_unet = unet(4)
write(out + "/sd8", UnetRoot(base_unet), TextRoot(text), 4)
write(out + "/sd8-inpaint", UnetRoot(unet(9)), TextRoot(text), 9)

# LoRA: rank 4 on the attention projections of the UNet and the text encoder's q/v.
rank = 4
unet_cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights=False)
text_cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
torch.manual_seed(5)
inject_adapter_in_model(unet_cfg, base_unet)
inject_adapter_in_model(text_cfg, text)
with torch.no_grad():
    for m in list(base_unet.modules()) + list(text.modules()):
        if hasattr(m, "lora_B") and "default" in getattr(m, "lora_B", {}):
            m.lora_A["default"].weight.normal_(0, 0.3)
            m.lora_B["default"].weight.normal_(0, 0.3)

unet_sd = get_peft_model_state_dict(base_unet)
text_sd = get_peft_model_state_dict(text)
peft_file = {"unet." + k: v.contiguous() for k, v in unet_sd.items()}
peft_file.update({"text_encoder.text_model." + k: v.contiguous() for k, v in text_sd.items()})
save_file(peft_file, out + "/lora-peft.safetensors")

# kohya: dots become underscores, lora_A -> lora_down, lora_B -> lora_up, alpha tensor.
# alpha = rank / 2 halves the scale, so B is doubled to describe the same update.
kohya = {}
for prefix, sd in (("lora_unet_", unet_sd), ("lora_te_text_model_", text_sd)):
    for k, v in sd.items():
        module, which = k.rsplit(".lora_", 1)
        key = prefix + module.replace(".", "_")
        if which.startswith("A"):
            kohya[key + ".lora_down.weight"] = v.contiguous()
            kohya[key + ".alpha"] = torch.tensor(rank / 2)
        else:
            kohya[key + ".lora_up.weight"] = (v * 2).contiguous()
save_file(kohya, out + "/lora-kohya.safetensors")


def merge(model):
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "merge") and hasattr(m, "lora_A") and "default" in m.lora_A:
                m.merge()
    # Strip the wrappers: replace every LoRA layer with its merged base layer.
    for name, m in list(model.named_modules()):
        for child_name, child in list(m.named_children()):
            if hasattr(child, "base_layer") and hasattr(child, "lora_A"):
                setattr(m, child_name, child.base_layer)
    return model


import tempfile
lora_dir = tempfile.mkdtemp()
write(lora_dir, UnetRoot(merge(base_unet)), TextRoot(merge(text)), 4)


# ---------------------------------------------------------------- references
def mean_encoder(path):
    """The encoder with its RandomNormalLike scale set to 0, so it returns the posterior mean."""
    import onnx, onnxruntime
    from onnx import helper
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type == "RandomNormalLike":
            scale = [a for a in n.attribute if a.name == "scale"]
            if scale: scale[0].f = 0.0
            else: n.attribute.append(helper.make_attribute("scale", 0.0))
    return OnnxRuntimeModel(onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]))


def build(cls, root):
    ort = lambda d: OnnxRuntimeModel(OnnxRuntimeModel.load_model(root + "/" + d + "/model.onnx", provider="CPUExecutionProvider"))
    return cls(vae_encoder=mean_encoder(root + "/vae_encoder/model.onnx"), vae_decoder=ort("vae_decoder"), text_encoder=ort("text_encoder"),
               tokenizer=CLIPTokenizer.from_pretrained(root + "/tokenizer"), unet=ort("unet"),
               scheduler=DDIMScheduler.from_pretrained(root + "/scheduler"), safety_checker=None,
               feature_extractor=None, requires_safety_checker=False)


image = Image.new("RGB", (64, 64)); px = image.load()
mask = Image.new("RGB", (64, 64)); mx = mask.load()
for y in range(64):
    for x in range(64):
        px[x, y] = ((x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256)
        mx[x, y] = (255, 255, 255) if 16 <= x < 44 and 20 <= y < 52 else (0, 0, 0)
image.save(out + "/init.png"); mask.save(out + "/mask.png")

prompt = "a red lighthouse at dawn"


class Euler(EulerDiscreteScheduler):
    # diffusers' ONNX pipeline multiplies a numpy array by this, which fails when it is a tensor.
    @property
    def init_noise_sigma(self):
        return float(super().init_noise_sigma)


def save(name, img):
    Image.fromarray((img * 255).round().astype(np.uint8)).save(f"{out}/ref-{name}.png")
    return (img * 255).round().astype(np.uint8)


t2i = dict(num_inference_steps=4, height=64, width=64, output_type="np")
base = save("t2i-ddim", build(OnnxStableDiffusionPipeline, out + "/sd8")(prompt, generator=np.random.RandomState(3), **t2i).images[0])
pipe = build(OnnxStableDiffusionPipeline, out + "/sd8")
pipe.scheduler = Euler.from_config(pipe.scheduler.config)
save("t2i-euler", pipe(prompt, generator=np.random.RandomState(3), **t2i).images[0])
lora = save("lora", build(OnnxStableDiffusionPipeline, lora_dir)(prompt, generator=np.random.RandomState(3), **t2i).images[0])
save("img2img", build(OnnxStableDiffusionImg2ImgPipeline, out + "/sd8")(
    prompt, image=image, strength=0.6, num_inference_steps=5, generator=np.random.RandomState(11), output_type="np").images[0])
save("inpaint", build(OnnxStableDiffusionInpaintPipeline, out + "/sd8-inpaint")(
    prompt, image=image, mask_image=mask, num_inference_steps=4, height=64, width=64,
    generator=np.random.RandomState(9), output_type="np").images[0])
shutil.rmtree(lora_dir)
print("lora differs from base in", int(np.sum(base != lora)), "of", base.size, "values")
