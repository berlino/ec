import torch.nn as nn

from neural_seq.decode import score_decode
from neural_seq.decoder import Decoder, MAX_PROGRAM_LENGTH
from neural_seq.encoder import LARCEncoder


class EncoderDecoder(nn.Module):
    """
    Program Synthesis model that takes in a feature vector (encoding) from which it outputs (decodes) a program. 

    Encoder:
        1. Encode IO examples with existing ARC CNN
        2. Encode NL description with existing T5 encoder

    Decoder:
        1. Use encoder output + partial program encoding as attention query, and NL description words / BERT encodings as keys and values
        2. Concatenate output from 1 with query from 1
        3. Use output from 2 as query and token embeddings as keys to get attention vector
    """

    def __init__(self, grammar, request, cuda, device, rnn_decode, use_nl=True, use_io=True, program_embedding_size=128, primitive_to_idx=None):
        super(EncoderDecoder, self).__init__()
        
        self.device = device
        self.rnn_decode = rnn_decode
        # there are three additional tokens, one for the start token input grid (tgridin) variable and the other for input 
        # variable of lambda expression (assumes no nested lambdas)
        print("Starting to load decoder")
        self.decoder = Decoder(embedding_size=program_embedding_size, grammar=grammar, request=request, cuda=cuda, device=device, max_program_length=MAX_PROGRAM_LENGTH, 
            encoderOutputSize=64, primitive_to_idx=primitive_to_idx, use_rnn=rnn_decode)
        print("Finished loading Decoder, starting to load Encoder")
        self.encoder = LARCEncoder(cuda=cuda, use_nl=use_nl, use_io=use_io, device=device)
        print("Finished loading Encoder")
    
        if cuda: self.cuda()

    def forward(self, io_grids, test_in, desc_tokens, mode, targets=None):
        
        encoderOutputs = self.encoder(io_grids, test_in, desc_tokens)
        batch_scores = score_decode(self.decoder, encoderOutputs, targets, self.rnn_decode, self.device)
        return batch_scores
