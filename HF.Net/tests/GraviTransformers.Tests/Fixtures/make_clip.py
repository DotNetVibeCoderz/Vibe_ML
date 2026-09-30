"""Fixtures for the CLIP and image-processor tests. Run from this folder: python make_clip.py

clip-tiny/          a random CLIPModel (2 layers, 32 wide, vocab 99, 32 px images in 8 px patches),
                    end token 98 - the rule current checkpoints use
clip-tiny-legacy/   the same weights, with eos_token_id 2 - OpenAI's checkpoints, pooled at the
                    highest token id
clip-reference.json torch float64 outputs for both
photo.png           a 150x100 test image
processor-*.bin     the slow (PIL) image processors' float32 output for it
"""
import json, os, shutil
import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor, ViTImageProcessor

torch.manual_seed(0)
config = CLIPConfig(
    text_config=dict(vocab_size=99, hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
                     max_position_embeddings=16, eos_token_id=98, bos_token_id=96, pad_token_id=1),
    vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8),
    projection_dim=16)
model = CLIPModel(config).eval()

# The default initialisation is so small that every layer is close to the identity; widen it, and give
# the norms and biases values of their own, so the test exercises every parameter.
with torch.no_grad():
    for name, p in model.named_parameters():
        if name == "logit_scale":
            continue
        p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or "layernorm" in name or "layrnorm" in name or "final_layer_norm.weight" in name:
            p.add_(1.0)

for folder, eos in (("clip-tiny", 98), ("clip-tiny-legacy", 2)):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    model.config.text_config.eos_token_id = eos
    model.save_pretrained(folder)

model = model.double()
reference = {}
texts = {
    # First end token at 2 (new rule); highest id at 2 as well for the legacy rule would hide a bug,
    # so the legacy sequence puts its highest id elsewhere.
    "new": [96, 5, 98, 17, 98, 1],
    "legacy": [0, 5, 97, 42, 2, 11],
}
pixels = torch.randn(1, 3, 32, 32, dtype=torch.float64)
reference["pixels"] = pixels.flatten().tolist()

for rule, ids in texts.items():
    model.config.text_config.eos_token_id = model.text_model.config.eos_token_id = 98 if rule == "new" else 2
    model.text_model.eos_token_id = 98 if rule == "new" else 2
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        text = model.text_model(input_ids=input_ids)
        features = model.text_projection(text.pooler_output)
    reference[f"{rule}_ids"] = ids
    reference[f"{rule}_hidden"] = text.last_hidden_state.flatten().tolist()
    reference[f"{rule}_features"] = features.flatten().tolist()

with torch.no_grad():
    vision = model.vision_model(pixel_values=pixels)
    image_features = model.visual_projection(vision.pooler_output)
    batch = torch.tensor([[96, 5, 98], [96, 17, 23, 98], [96, 60, 61, 62, 98]][0:1])
    logits = []
    model.text_model.eos_token_id = 98
    for ids in ([96, 5, 98], [96, 17, 23, 98], [96, 60, 61, 62, 98]):
        t = model.text_projection(model.text_model(input_ids=torch.tensor([ids])).pooler_output)
        i = image_features / image_features.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        logits.append((model.logit_scale.exp() * (i @ t.T)).item())

reference["vision_hidden"] = vision.last_hidden_state.flatten().tolist()
reference["image_features"] = image_features.flatten().tolist()
reference["logit_texts"] = [[96, 5, 98], [96, 17, 23, 98], [96, 60, 61, 62, 98]]
reference["logits"] = logits
json.dump(reference, open("clip-reference.json", "w"))

# ---------------------------------------------------------------- image processors
image = Image.new("RGB", (150, 100))
px = image.load()
rng = np.random.RandomState(4)
noise = rng.randint(0, 256, (100, 150, 3))
for y in range(100):
    for x in range(150):
        # Smooth gradients plus noise, so both the averaging and the edges of the kernel matter.
        px[x, y] = tuple(int((v + g) % 256) for v, g in zip(noise[y, x] // 3, ((x * 5) % 256, (y * 7) % 256, ((x + y) * 3) % 256)))
image.save("photo.png")

clip = CLIPImageProcessor(size={"shortest_edge": 48}, crop_size={"height": 40, "width": 40})
vit = ViTImageProcessor(size={"height": 40, "width": 40})
for name, processor in (("clip", clip), ("vit", vit)):
    values = processor(images=image, return_tensors="np")["pixel_values"][0].astype(np.float32)
    values.tofile(f"processor-{name}.bin")
    processor.save_pretrained(f"processor-{name}")
    print(name, type(processor).__name__, values.shape)
