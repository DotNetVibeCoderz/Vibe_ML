"""Fixtures for the TrOCR tests. Run from this folder: python make_trocr.py [--real]

trocr-tiny/            a random VisionEncoderDecoderModel: distilled DeiT encoder (2 layers, 32 wide, 32 px
                       images in 8 px patches, so 2 + 16 tokens) + TrOCR decoder (2 layers, 24 wide, relu,
                       learned positions, scale_embedding, layernorm_embedding, its own output layer), with a
                       tiny XLM-R SentencePiece tokenizer (nmt_nfkc, so it carries a Precompiled charsmap -
                       transformers cannot read an identity-normalised model)
trocr-tiny-vit/        the other variant: ViT encoder without q/k/v bias + decoder with sinusoidal positions,
                       gelu, no embedding norm or scale, output layer tied to the token table; no tokenizer
                       (the charsmap alone is 240 KB, so it is kept once)
trocr-reference.json   torch float64 outputs for both: pixels, encoder states, decoder logits, greedy and
                       beam-search ids and their decoded text, tokenizer ids
trocr-line.png         the line image the tiny references were computed from

--real also writes, from microsoft/trocr-small-printed and microsoft/trocr-base-printed (downloaded to
the shared Hugging Face cache):
trocr-lines/line*.png           three rendered text lines (Arial 32)
trocr-printed-reference.json    per model and line: float64 pixel/encoder/first-step-logit samples, and the
                                generated ids and text from float32 (as users run it) and float64

Sampling: transformers' outputs are too large to keep whole (578 x 384 encoder states, 64044 logits), so the
reference stores every 997th encoder value, every 509th logit, the ten largest logits and every 1009th pixel.
"""
import io, json, os, shutil, sys
import numpy as np
import sentencepiece as spm
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (DeiTConfig, DeiTModel, GenerationConfig, TrOCRConfig, TrOCRForCausalLM, ViTConfig, ViTModel,
                          VisionEncoderDecoderModel, XLMRobertaTokenizer, RobertaTokenizer,
                          DeiTImageProcessorPil, ViTImageProcessorPil)

LINES = ["Net revenue rose 12% to $4.3 billion", "Laba bersih Rp 1.234 miliar", "Operating margin: 18.5% (Q3 2025)"]


def render(text, size=32, font_path="C:/Windows/Fonts/arial.ttf"):
    font = ImageFont.truetype(font_path, size) if os.path.exists(font_path) else ImageFont.load_default(size)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGB", (right - left + 24, bottom - top + 20), "white")
    ImageDraw.Draw(image).text((12 - left, 10 - top), text, fill="black", font=font)
    return image


def tolist(t):
    return t.detach().double().flatten().tolist()


# ---------------------------------------------------------------- tiny tokenizer
corpus = LINES + ["Total assets increased to 9,876 million", "Pendapatan naik 7% menjadi Rp 5,6 triliun",
                  "EBITDA fell 3.2% year on year", "Cash and equivalents: $812 million"]
model_file = io.BytesIO()
spm.SentencePieceTrainer.train(sentence_iterator=iter(corpus * 20), model_writer=model_file, vocab_size=120,
                               model_type="bpe", character_coverage=1.0,
                               minloglevel=2)


def save_tokenizer(folder):
    with open(os.path.join(folder, "sentencepiece.bpe.model"), "wb") as f:
        f.write(model_file.getvalue())
    with open(os.path.join(folder, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "XLMRobertaTokenizer", "model_max_length": 512}, f)


os.makedirs("trocr-tokenizer-tmp", exist_ok=True)
save_tokenizer("trocr-tokenizer-tmp")
tokenizer = XLMRobertaTokenizer.from_pretrained("trocr-tokenizer-tmp")
shutil.rmtree("trocr-tokenizer-tmp")
vocab = len(tokenizer)
print("tiny vocabulary", vocab)

# ---------------------------------------------------------------- tiny models
torch.manual_seed(0)
variants = {
    "trocr-tiny": (
        DeiTModel(DeiTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37,
                             image_size=32, patch_size=8, hidden_act="gelu"), add_pooling_layer=False),
        TrOCRConfig(vocab_size=vocab, d_model=24, decoder_layers=2, decoder_attention_heads=4, decoder_ffn_dim=40,
                    activation_function="relu", max_position_embeddings=32, cross_attention_hidden_size=32,
                    scale_embedding=True, layernorm_embedding=True, use_learned_position_embeddings=True,
                    tie_word_embeddings=False, is_decoder=True, add_cross_attention=True),
        DeiTImageProcessorPil(size={"height": 32, "width": 32}, resample=3, do_center_crop=False,
                              image_mean=[0.5] * 3, image_std=[0.5] * 3),
    ),
    "trocr-tiny-vit": (
        ViTModel(ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37,
                           image_size=32, patch_size=8, qkv_bias=False, hidden_act="gelu"), add_pooling_layer=False),
        TrOCRConfig(vocab_size=vocab, d_model=24, decoder_layers=2, decoder_attention_heads=4, decoder_ffn_dim=40,
                    activation_function="gelu", max_position_embeddings=32, cross_attention_hidden_size=32,
                    scale_embedding=False, layernorm_embedding=False, use_learned_position_embeddings=False,
                    tie_word_embeddings=True, is_decoder=True, add_cross_attention=True),
        ViTImageProcessorPil(size={"height": 32, "width": 32}, resample=2, image_mean=[0.5] * 3, image_std=[0.5] * 3),
    ),
}

line = render(LINES[0], size=16)
line.save("trocr-line.png")
reference = {}

for name, (encoder, decoder_config, processor) in variants.items():
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=TrOCRForCausalLM(decoder_config)).eval()
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = 1
    model.config.eos_token_id = 2
    model.generation_config = GenerationConfig(decoder_start_token_id=2, bos_token_id=0, eos_token_id=2, pad_token_id=1, use_cache=False)

    # The default initialisation is so small that every layer is close to the identity; widen it, and give
    # the norms values of their own, so the test exercises every parameter. Then make the end token as
    # likely as a token greedy decoding picks a few steps in, so the texts end on their own and the
    # end-token paths run. Seeds are tried in order until the text has at least three distinct tokens.
    probe = processor(images=line, return_tensors="pt").pixel_values
    for seed in range(50):
        torch.manual_seed(seed)
        with torch.no_grad():
            for parameter_name, p in model.named_parameters():
                p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
                if "norm" in parameter_name and parameter_name.endswith("weight"):
                    p.add_(1.0)
                # Small token vectors, so the next token depends on the image and the position and not
                # only on the token before it - otherwise a random decoder falls into a loop at once.
                if "embed_tokens" in parameter_name:
                    p.mul_(0.2)
            head = model.decoder.get_output_embeddings().weight
            ids = model.generate(probe, max_new_tokens=12)[0].tolist()
            if len(set(ids[1:9])) < 3:
                continue
            chosen = None
            for step in range(2, 10):
                if ids[step] in ids[1:step] or ids[step] == 2:
                    continue
                original = head[2].clone()
                for scale in (1.0, 1.02, 1.05, 1.1):
                    head[2] = head[ids[step]] * scale
                    ended = model.generate(probe, max_new_tokens=12)[0].tolist()
                    if ended[-1] == 2 and 5 <= len(ended) <= 13:
                        chosen = (step, scale, ended)
                        break
                    head[2] = original
                if chosen:
                    break
            if chosen:
                break
    print(name, "seed", seed, "end token copied from step", chosen)

    if os.path.exists(name):
        shutil.rmtree(name)
    model.save_pretrained(name)
    processor.save_pretrained(name)
    if name == "trocr-tiny":
        save_tokenizer(name)

    model = model.double()
    pixels = processor(images=line, return_tensors="pt").pixel_values.double()
    with torch.no_grad():
        encoded = model.encoder(pixel_values=pixels).last_hidden_state
        steps = torch.tensor([[2, 17, 5, 1, 40]])   # a pad token among them: sinusoidal positions skip it
        logits = model(pixel_values=pixels, decoder_input_ids=steps).logits
        greedy = model.generate(pixels, max_new_tokens=12)
        beam = model.generate(pixels, max_new_tokens=8, num_beams=3)
        beam_penalised = model.generate(pixels, max_new_tokens=8, num_beams=4, length_penalty=0.5)
    reference[name] = {
        "pixels": tolist(pixels),
        "encoder": tolist(encoded),
        "encoder_shape": list(encoded.shape[1:]),
        "decoder_ids": steps[0].tolist(),
        "logits": tolist(logits),
        "greedy": greedy[0].tolist(),
        "greedy_text": tokenizer.batch_decode(greedy, skip_special_tokens=True)[0],
        "beam": beam[0].tolist(),
        "beam_penalised": beam_penalised[0].tolist(),
    }
    print(name, "greedy", greedy[0].tolist(), "beam", beam[0].tolist(), "beam lp", beam_penalised[0].tolist())

reference["tokenizer"] = {text: tokenizer(text).input_ids for text in corpus}
reference["tokenizer_decoded"] = {text: tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True) for text in corpus}
json.dump(reference, open("trocr-reference.json", "w"))

if "--real" not in sys.argv:
    sys.exit(0)

# ---------------------------------------------------------------- the published checkpoints
os.makedirs("trocr-lines", exist_ok=True)
images = []
for i, text in enumerate(LINES):
    image = render(text)
    image.save(f"trocr-lines/line{i}.png")
    images.append(image)

printed = {"lines": LINES}
for repo, tokenizer_class, processor_class in (
        ("microsoft/trocr-small-printed", XLMRobertaTokenizer, DeiTImageProcessorPil),
        ("microsoft/trocr-base-printed", RobertaTokenizer, ViTImageProcessorPil)):
    tok = tokenizer_class.from_pretrained(repo)
    processor = processor_class.from_pretrained(repo)
    model32 = VisionEncoderDecoderModel.from_pretrained(repo).eval()
    model64 = VisionEncoderDecoderModel.from_pretrained(repo).eval().double()
    entries = []
    for image in images:
        pixels = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids32 = model32.generate(pixels)
            beam32 = model32.generate(pixels, num_beams=4)
            encoded32 = model32.encoder(pixel_values=pixels).last_hidden_state.double().flatten()
            logits32 = model32(pixel_values=pixels, decoder_input_ids=torch.tensor([[2]])).logits.double().flatten()
            p64 = pixels.double()
            encoded = model64.encoder(pixel_values=p64).last_hidden_state.flatten()
            logits = model64(pixel_values=p64, decoder_input_ids=torch.tensor([[2]])).logits.flatten()
            ids64 = model64.generate(p64)
        top = torch.topk(logits, 10)
        flat_pixels = pixels.double().flatten()
        entries.append({
            "pixel_sum": flat_pixels.sum().item(),
            "pixel_samples": flat_pixels[::1009].tolist(),
            "encoder_shape": list(model64.encoder(pixel_values=p64).last_hidden_state.shape[1:]),
            "encoder_samples": encoded[::997].tolist(),
            "encoder32_samples": encoded32[::997].tolist(),
            "encoder32_vs_64_max": (encoded32 - encoded).abs().max().item(),
            "logit_samples": logits[::509].tolist(),
            "logit32_samples": logits32[::509].tolist(),
            "logits32_vs_64_max": (logits32 - logits).abs().max().item(),
            "top_ids": top.indices.tolist(),
            "top_logits": top.values.tolist(),
            "ids": ids32[0].tolist(),
            "text": tok.batch_decode(ids32, skip_special_tokens=True)[0],
            "ids64": ids64[0].tolist(),
            "text64": tok.batch_decode(ids64, skip_special_tokens=True)[0],
            "beam4_ids": beam32[0].tolist(),
            "beam4_text": tok.batch_decode(beam32, skip_special_tokens=True)[0],
            "tokenizer_ids": tok(LINES[len(entries)]).input_ids,
        })
        print(repo, entries[-1]["text"], entries[-1]["ids"], "f32 vs f64:", entries[-1]["encoder32_vs_64_max"], entries[-1]["logits32_vs_64_max"])
    printed[repo] = entries

json.dump(printed, open("trocr-printed-reference.json", "w"))
