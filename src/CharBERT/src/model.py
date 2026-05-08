import math

from typing import Optional, Tuple



import torch

from torch import nn





class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 2048):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe[:, : x.size(1)]





def _build_char_lookup_table(vocab_size: int, char_vocab: str) -> Tuple[torch.Tensor, int]:

    if char_vocab not in {"hex", "ascii"}:

        raise ValueError(f"unsupported char_vocab: {char_vocab}")



    token_strings = []

    for token_id in range(vocab_size):

        if char_vocab == "hex":

            token_strings.append(format(token_id, "03X"))

        else:

            if token_id < 256 and 32 <= token_id <= 126:

                token_strings.append(chr(token_id))

            elif token_id < 256:

                token_strings.append(".")

            else:

                token_strings.append(f"<{token_id}>")



    alphabet = sorted({ch for text in token_strings for ch in text})

    char_to_id = {ch: i + 1 for i, ch in enumerate(alphabet)}

    max_char_len = max(len(s) for s in token_strings)

    table = torch.zeros((vocab_size, max_char_len), dtype=torch.long)

    for token_id, token_text in enumerate(token_strings):

        encoded = [char_to_id[ch] for ch in token_text]

        table[token_id, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)

    return table, len(alphabet) + 1





class CharBERT(nn.Module):

    def __init__(

        self,

        vocab_size: int,

        hidden_size: int,

        num_layers: int,

        num_heads: int,

        dropout: float,

        num_labels: int,

        max_len: int,

        mode: str = "legacy",

        char_vocab: str = "hex",

        char_emb_dim: int = 32,

        char_cnn_channels: int = 64,

        char_fusion: str = "gated",

        char_fusion_layers: str = "all",

    ):

        super().__init__()

        if mode not in {"legacy", "charaware"}:

            raise ValueError(f"unsupported mode: {mode}")

        if char_fusion not in {"gated", "add", "concat"}:

            raise ValueError(f"unsupported char_fusion: {char_fusion}")

        if char_fusion_layers not in {"first", "last", "all"}:

            raise ValueError(f"unsupported char_fusion_layers: {char_fusion_layers}")



        self.mode = mode

        self.char_fusion = char_fusion

        self.char_fusion_layers = char_fusion_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=vocab_size - 3)                                

        self.pos_encoder = PositionalEncoding(hidden_size, max_len)

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_size,

            nhead=num_heads,

            dim_feedforward=hidden_size * 4,

            dropout=dropout,

            batch_first=True,

        )

        if mode == "legacy":

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.encoder_layers = None

        else:

            self.encoder = None

            self.encoder_layers = nn.ModuleList(

                [

                    nn.TransformerEncoderLayer(

                        d_model=hidden_size,

                        nhead=num_heads,

                        dim_feedforward=hidden_size * 4,

                        dropout=dropout,

                        batch_first=True,

                    )

                    for _ in range(num_layers)

                ]

            )



            char_id_table, char_vocab_size = _build_char_lookup_table(vocab_size=vocab_size, char_vocab=char_vocab)

            self.register_buffer("char_id_table", char_id_table)

            self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)

            self.char_conv = nn.Conv1d(char_emb_dim, char_cnn_channels, kernel_size=3, padding=1)

            self.char_proj = nn.Linear(char_cnn_channels, hidden_size)

            self.char_activation = nn.GELU()



            if char_fusion == "gated":

                self.fusion_gate = nn.Sequential(

                    nn.Linear(hidden_size * 2, hidden_size),

                    nn.ReLU(),

                    nn.Linear(hidden_size, hidden_size),

                    nn.Sigmoid(),

                )

            elif char_fusion == "concat":

                self.concat_proj = nn.Linear(hidden_size * 2, hidden_size)



        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, num_labels)



    def _encode_char_features(self, input_ids: torch.Tensor) -> torch.Tensor:

        char_ids = self.char_id_table[input_ids]           

        batch_size, seq_len, char_len = char_ids.shape

        char_emb = self.char_embedding(char_ids).reshape(batch_size * seq_len, char_len, -1)

        char_emb = char_emb.transpose(1, 2)             

        conv_out = self.char_activation(self.char_conv(char_emb))

        pooled = conv_out.max(dim=-1).values

        return self.char_proj(pooled).reshape(batch_size, seq_len, -1)



    def _fuse_token_char(self, token_hidden: torch.Tensor, char_hidden: torch.Tensor) -> torch.Tensor:

        if self.char_fusion == "add":

            return token_hidden + char_hidden

        if self.char_fusion == "concat":

            return self.concat_proj(torch.cat([token_hidden, char_hidden], dim=-1))

        gate = self.fusion_gate(torch.cat([token_hidden, char_hidden], dim=-1))

        return gate * token_hidden + (1.0 - gate) * char_hidden



    def _need_fuse(self, layer_idx: int, layer_count: int) -> bool:

        if self.char_fusion_layers == "all":

            return True

        if self.char_fusion_layers == "first":

            return layer_idx == 0

        return layer_idx == layer_count - 1



    def encode_tokens(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):

        x = self.embedding(input_ids)

        x = self.pos_encoder(x)

        if attention_mask is not None:

            x = x * attention_mask.unsqueeze(-1)

            pad_mask = attention_mask == 0

        else:

            pad_mask = None



        if self.mode == "legacy":

            enc = self.encoder(x, src_key_padding_mask=pad_mask)

            return enc, pad_mask



        char_hidden = self._encode_char_features(input_ids)

        hidden = x

        layer_count = len(self.encoder_layers)

        for layer_idx, layer in enumerate(self.encoder_layers):

            if self._need_fuse(layer_idx, layer_count):

                hidden = self._fuse_token_char(hidden, char_hidden)

            hidden = layer(hidden, src_key_padding_mask=pad_mask)

        return hidden, pad_mask



    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):

        x, _ = self.encode_tokens(input_ids, attention_mask=attention_mask)

        pooled = x[:, 0]          

        logits = self.classifier(self.dropout(pooled))

        return logits





def build_model(cfg, num_labels: int):

    return CharBERT(

        vocab_size=cfg.vocab_size,

        hidden_size=cfg.hidden_size,

        num_layers=cfg.num_layers,

        num_heads=cfg.num_heads,

        dropout=cfg.dropout,

        num_labels=num_labels,

        max_len=cfg.max_len,

        mode=getattr(cfg, "mode", "legacy"),

        char_vocab=getattr(cfg, "char_vocab", "hex"),

        char_emb_dim=getattr(cfg, "char_emb_dim", 32),

        char_cnn_channels=getattr(cfg, "char_cnn_channels", 64),

        char_fusion=getattr(cfg, "char_fusion", "gated"),

        char_fusion_layers=getattr(cfg, "char_fusion_layers", "all"),

    )



