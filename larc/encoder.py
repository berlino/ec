import torch
import torch.nn as nn
from transformers import BertModel

class LARCEncoder(nn.Module):
    """
    context encoder of LARC task
    assumes 3 IO examples, where each is 30x30. assumes test input is 30x30. assumes description is tokenized by LM.
    https://files.slack.com/files-pri/T01044K0LBZ-F02D2MGQF62/screen_shot_2021-09-02_at_1.56.03_pm.png
    TODO:
        - better way to handle different grid sizes than padding with special token
        - make LM changeable at initialization
    """
    def __init__(self, cuda, device, use_nl=True, use_io=True):
        super(LARCEncoder, self).__init__()
        
        self.device = device
        self.use_nl = use_nl
        self.use_io = use_io

        # grid encoder
        # 30x30x11 --> 256
        self.encoder = nn.Sequential(
            nn.Conv2d(11, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.Flatten(),
        )
        
        if self.use_io:
            # input vs. output embedding
            # 256 --> 128
            self.in_encoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.out_encoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
            )

            # example embedding
            # 256 --> 64
            self.ex_encoder = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
            )

        # test input embedding
        # 256 --> 64
        self.test_in_embedding = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        if self.use_nl:
            # natural language description encoding
            # nl --> 64
            self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir=".cache/")
            self.bert.requires_grad_(False)
            self.bert_resize = nn.Sequential(
                nn.Linear(768, 64),
                nn.ReLU(),
            )

        # transformer:
        # seq_length x batch_size x d_model --> seq_length x batch_size x d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        if cuda: self.cuda()


    def forward(self, io_grids, test_in, desc_tokens):
        # run grids through encoders
        transformer_input = []
        if self.use_io:
            for io_in, io_out in io_grids:
                io_in = self.in_encoder(self.encoder(io_in))
                io_out = self.out_encoder(self.encoder(io_out))
                io = self.ex_encoder(torch.cat((io_in, io_out), dim=-1))
            transformer_input.append(io)

        # run test input grid through encoder
        transformer_input.append(self.test_in_embedding(self.encoder(test_in)))

        if self.use_nl:
            # run through BERT
            transformer_input.append(self.bert_resize(self.bert(**desc_tokens)['pooler_output']))

        # stack all inputs
        t_in = torch.stack(transformer_input, dim=1)
        # change shape so that t_in works with batch_first=False
        t_in = t_in.transpose(0, 1)
        # seq_length x batch_size x d_model --> seq_length x batch_size x d_model
        t_out = self.transformer(t_in)

        # TODO: ask evan about changing
        # aggregate transformers vector outputs into a single vector by taking max operation
        t_out = torch.max(t_out, dim=0).values

        # new shape is: batch_size x embed_dim
        return t_out
