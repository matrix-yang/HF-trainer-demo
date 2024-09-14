import timm
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m',
                                       pretrained=True, img_size=384,
                                       pretrained_cfg_overlay=dict(
                                           file='/huawei-data/FM/checkpoints/timm/vit_large_patch14_reg4_dinov2.lvd142m/pytorch_model.bin')
                                       , )
        bert_path = '/huawei-data/FM/checkpoints/hfl/chinese-roberta-wwm-ext'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.llm = BertModel.from_pretrained(bert_path)
        self.cross_atten = BiMultiHeadAttention(v_dim=1024, l_dim=768, embed_dim=1024, num_heads=8)
        self.cls_head = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pix_values, input_dict):
        # 步骤4: 实现前向传播
        intermediate1 = self.model.forward_intermediates(pix_values, indices=[-2], intermediates_only=True)
        # [16, 729, 1024]
        vis_feat = intermediate1[0].flatten(2).permute(0, 2, 1)
        text_features = self.llm(**input_dict)
        last_hidden_state = text_features.last_hidden_state
        # pooler_output=text_features.pooler_output*0
        # print(last_hidden_state.shape,pooler_output.shape)
        # print(last_hidden_state[:,0,:]==pooler_output)
        fused_text_features = self.cross_atten(vis_feat, last_hidden_state)
        cls = self.cls_head(fused_text_features[:, 0])
        return self.sigmoid(cls).flatten()


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        # self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        # self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.values_l_proj.weight)
        # self.values_l_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.out_v_proj.weight)
        # self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        # value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1,
                      self.head_dim)  # (bs * 8, -1, embed_dim//8)
        query_states = self._shape(query_states, tgt_len, bsz).view(
            *proj_shape)  # (bs * 8, seq_len_img, embed_dim//8)
        key_states = key_states.view(
            *proj_shape)  # (bs * 8, seq_len_text, embed_dim//8)
        value_v_states = value_v_states.view(*proj_shape)
        # value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1,
                                                                    2))  # (bs * 8, seq_len_img, seq_len_text)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        # attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        # attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        # if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
        #     )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_l