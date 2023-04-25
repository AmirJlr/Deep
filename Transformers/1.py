import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 256
        self.heads = heads  # 8
        self.head_dim = embed_size // heads  # 32

        assert (self.head_dim * heads == embed_size), "Embed size must be div by heads"

        # first linear layers for V, K, Q
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # map to embed size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]  # number of training examples

        # corresponding to sentence length...
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Matmul Q, K
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        # queries shape : (N, query_len, heads, head_dim)
        # keys shape : (N, key_len, heads, head_dim)
        # energy shape : N, heads, query_len(~target source sentence), key_len(~ source sentence))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(-1e20))

        # Softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape : n, head, query_len, key_len
        # values shape : N, value_len, heads, head_dim
        # after einsum : (N, query_len, heads, head_dim) then flatten last two dimension

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # mapping into some more nodes(in paper:4)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention, query))  # skip connection
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)  # skip connection
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size,
                                 heads,
                                 dropout=dropout,
                                 forward_expansion=forward_expansion
                                 )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            # in the encoder value, key, query are all the same
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):  # src_mask : optional, trg_mask : essential
        attention = self.attention(x, x, x, trg_mask)

        query = self.dropout(self.norm(attention + x))  # skip connection
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)  # value, key in decoder block

        out = self.fc_out(x)
        return out


# Putting Together :
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, src_pad_inx, target_pad_inx, embed_size=256, num_layers=6,
                 forward_expansion=4, heads=8, dropout=0, device="cuda", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
        )

        self.decoder = Decoder(
            target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length
        )

        self.src_pad_inx = src_pad_inx
        self.target_pad_inx = target_pad_inx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_inx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape

        # triangular Matrix
        target_mask = torch.tril(torch.ones(target_len, target_len)).expand(N, 1, target_len, target_len)
        return target_mask.to(self.device)

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)
        return out
