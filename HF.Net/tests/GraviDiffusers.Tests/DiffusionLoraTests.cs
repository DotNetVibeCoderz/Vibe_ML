using Gravicode.HFNet.GraviDiffusers;
using Xunit;

namespace Gravicode.HFNet.GraviDiffusers.Tests;

/// <summary>Reading LoRA key layouts, and finding the module an ONNX node came from.</summary>
public sealed class DiffusionLoraTests
{
    [Theory]
    [InlineData("unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight",
        "unet", "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q", "down")]
    [InlineData("unet.mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.weight",
        "unet", "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0", "up")]
    [InlineData("text_encoder.text_model.encoder.layers.3.self_attn.v_proj.lora.down.weight",
        "text_encoder", "text_model.encoder.layers.3.self_attn.v_proj", "down")]
    [InlineData("unet.up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor.to_out_lora.up.weight",
        "unet", "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0", "up")]
    [InlineData("lora_unet_down_blocks_0_attentions_1_proj_in.lora_down.weight",
        "unet", "kohya:down_blocks_0_attentions_1_proj_in", "down")]
    [InlineData("lora_te_text_model_encoder_layers_0_mlp_fc1.alpha",
        "text_encoder", "kohya:text_model_encoder_layers_0_mlp_fc1", "alpha")]
    public void Every_layout_resolves_to_a_component_and_module(string key, string component, string module, string role)
    {
        Assert.Equal((component, module, role), DiffusionLora.Classify(key));
    }

    [Theory]
    [InlineData("unet.conv_in.weight")]
    [InlineData("lora_unet_down_blocks_0_attentions_1_proj_in.dora_scale")]
    [InlineData("vae.decoder.conv_in.weight")]
    public void A_key_outside_an_adapter_pair_is_ignored(string key)
    {
        Assert.Null(DiffusionLora.Classify(key));
    }

    [Theory]
    [InlineData("/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q/MatMul", "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q")]
    [InlineData("/text_model/encoder/layers.0/self_attn/q_proj/MatMul", "text_model.encoder.layers.0.self_attn.q_proj")]
    [InlineData("/conv_in/Conv", "conv_in")]
    [InlineData("MatMul_12", "")]
    public void A_node_name_gives_the_module_path(string node, string module)
    {
        Assert.Equal(module, DiffusionLora.ModuleOf(node));
    }
}
