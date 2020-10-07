import torch
import tensorflow as tf


def format_attention(attention):
    assert "torch" in str(type(attention[0])) or "tensorflow" in str(type(attention[0]))
    framework = "torch" if "torch" in str(type(attention[0])) else "tf"
    
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        l_att = layer_attention.squeeze(0) if framework == "torch" else layer_attention[0]
        # squeezed.append(layer_attention.squeeze(0))
        squeezed.append(l_att)

    f_tensor = torch.stack(squeezed) if framework == "torch" else tf.stack(squeezed)
    # num_layers x num_heads x seq_len x seq_len
    return f_tensor


def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]
