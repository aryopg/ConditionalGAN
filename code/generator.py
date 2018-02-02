import sys
import time
import datetime
import logging
import cPickle

import numpy as np
import random

import torch

# import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0

class GeneratorEncDec(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDec, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True)
        self.decoder_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, motion):
        # ENCODER BLOCK
        # hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden

        embedded = self.encoder_embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.encoder_gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        encoder_hidden = torch.unsqueeze(hidden.permute(1,0,2)[:,-1,:], 1)

        # DECODER BLOCK
        # initially feed with [SOS] idx
        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)
        SOS_token = autograd.Variable(SOS_token)
        y_embedded = self.decoder_embeddings(SOS_token)
        y_embedded = self.dropout(y_embedded)
        # self.dropout(y_embedded)

        attn_weights = F.softmax(self.attn(torch.cat((y_embedded, encoder_hidden), 2)))
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((y_embedded, attn_applied), 2)
        output = self.attn_combine(output)

        decoder_hidden = encoder_hidden

        # decoder_output_full = []
        # for i in range(self.claim_length):
        #     decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
        #     decoder_output_full.append(decoder_output)
        # decoder_output = torch.cat(decoder_output_full, 1)


        output = output.permute(1,0,2).squeeze()

        decoder_output = output
        decoder_hidden = decoder_hidden.permute(1,0,2).squeeze()

        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.hidden_dim))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        # decoder_output_full = []
        for i in range(self.claim_length):
            decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
            decoder_output = decoder_hidden
            # decoder_output_full.append(decoder_output)
            decoder_output_full[i, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        gen_output = self.softmax(self.out(decoder_output_full))

        return gen_output

class GeneratorEncDecTeacherForcing(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcing, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True)
        self.decoder_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        embedded = self.encoder_embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.encoder_gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        encoder_hidden = torch.unsqueeze(hidden.permute(1,0,2)[:,-1,:], 1)

        return encoder_output, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        y_embedded = self.decoder_embeddings(SOS_token)
        y_embedded = self.dropout(y_embedded)

        attn_weights = F.softmax(self.attn(torch.cat((y_embedded, encoder_hidden), 2)))
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((y_embedded, attn_applied), 2)
        output = self.attn_combine(output)
        decoder_hidden = encoder_hidden

        output = output.permute(1,0,2).squeeze()
        decoder_output = output
        decoder_hidden = decoder_hidden.permute(1,0,2).squeeze()

        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.hidden_dim))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            target = target_output.permute(1,0)
            for i in range(self.claim_length):
                decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
                decoder_output_full[i, :, :] = decoder_hidden
                decoder_output = self.decoder_embeddings(target[i-1])  # Teacher forcing
                decoder_output = self.dropout(decoder_output)

        else:
            for i in range(self.claim_length):
                decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
                decoder_output = decoder_hidden
                decoder_output_full[i, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        gen_output = self.softmax(self.out(decoder_output_full))

        return gen_output

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)
        SOS_token = autograd.Variable(SOS_token)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

class GeneratorEncDecTeacherForcingV2(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcingV2, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True)
        self.decoder_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        embedded = self.encoder_embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.encoder_gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        encoder_hidden = torch.unsqueeze(hidden.permute(1,0,2)[:,-1,:], 1)

        return encoder_output, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        y_embedded = self.decoder_embeddings(SOS_token)
        y_embedded = self.dropout(y_embedded)

        attn_weights = F.softmax(self.attn(torch.cat((y_embedded, encoder_hidden), 2)))
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((y_embedded, attn_applied), 2)
        output = self.attn_combine(output)
        decoder_hidden = encoder_hidden

        output = output.permute(1,0,2).squeeze()
        decoder_input = output
        decoder_hidden = decoder_hidden.permute(1,0,2).squeeze()

        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.vocab_size))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            target = target_output.permute(1,0)
            for i in range(self.claim_length):
                decoder_hidden = self.decoder_gru_cell(decoder_input, decoder_hidden)
                decoder_output = self.softmax(self.out(decoder_hidden))
                decoder_output_full[i, :, :] = decoder_output
                decoder_input = self.decoder_embeddings(target[i-1])  # Teacher forcing
                decoder_input = self.dropout(decoder_input)

        else:
            for i in range(self.claim_length):
                decoder_input = decoder_input.squeeze()
                decoder_hidden = self.decoder_gru_cell(decoder_input, decoder_hidden)
                decoder_output = self.softmax(self.out(decoder_hidden))
                topv, topi = decoder_output.data.topk(1)
                # ni = topi#[0][0]
                # print("ni: ", ni)
                # decoder_input_v = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input_v = autograd.Variable(topi)
                decoder_input_v = decoder_input_v.cuda() if use_cuda else decoder_input_v
                decoder_input = self.decoder_embeddings(decoder_input_v)
                decoder_input = self.dropout(decoder_input)
                decoder_output_full[i, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        # gen_output = self.softmax(self.out(decoder_output_full))

        return decoder_output_full

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)
        SOS_token = autograd.Variable(SOS_token)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

class GeneratorEncDecTeacherForcingV3(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcingV3, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size
        self.vocab_size = vocab_size

        self.selu = nn.SELU()

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim)#, n_layers, dropout=0.5, batch_first=True)

        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        motion = motion.permute(1,0)
        encoder_outputs = autograd.Variable(torch.zeros(self.motion_length, self.hidden_dim))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        for idx, motion_word in enumerate(motion):
            if idx == 0:
                encoder_hidden = self.initHidden()
            embedded = self.encoder_embeddings(motion_word).view(1, 1, -1)
            encoder_output, encoder_hidden = self.encoder_gru(embedded, encoder_hidden) #[batch_size, seq_length, hidden_dim]
            encoder_outputs[idx] = encoder_output[0][0]
        return encoder_outputs, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.vocab_size))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        target = target_output.permute(1,0)

        for idx in range(self.claim_length):
            if idx == 0:
                decoder_input = SOS_token
                decoder_hidden = encoder_hidden
            embedded = self.decoder_embeddings(decoder_input).view(1, 1, -1)
            embedded = self.dropout(embedded)

            attn_weights = F.softmax(self.attn(torch.cat((embedded[0], decoder_hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)
            output = self.selu(output)

            use_teacher_forcing = True# if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                decoder_output_full[idx, :, :] = decoder_output
                decoder_input = target[idx-1]  # Teacher forcing

            else:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                # decoder_input_v = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                # print decoder_input
                decoder_output_full[idx, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        # gen_output = self.softmax(self.out(decoder_output_full))

        return decoder_output_full

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        SOS_token = autograd.Variable(SOS_token)
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

    def initHidden(self):
        result = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class GeneratorEncDecTeacherForcingV4(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcingV4, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim)#, n_layers, dropout=0.5, batch_first=True)

        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        motion = motion.permute(1,0)
        encoder_outputs = autograd.Variable(torch.zeros(self.motion_length, self.batch_size, self.hidden_dim))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        for idx, motion_word in enumerate(motion):
            if idx == 0:
                encoder_hidden = self.initHidden()
            embedded = self.encoder_embeddings(motion_word).view(1, self.batch_size, -1)
            encoder_output, encoder_hidden = self.encoder_gru(embedded, encoder_hidden) #[batch_size, seq_length, hidden_dim]
            encoder_outputs[idx] = encoder_output[0]
        return encoder_outputs, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.vocab_size))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        target = target_output.permute(1,0)
        encoder_output = encoder_output.permute(1,0,2)

        for idx in range(self.claim_length):
            if idx == 0:
                decoder_input = SOS_token
                decoder_hidden = encoder_hidden
            embedded = self.decoder_embeddings(decoder_input).view(1, self.batch_size, -1)
            embedded = self.dropout(embedded)

            attn_weights = F.softmax(self.attn(torch.cat((embedded[0], decoder_hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_output)
            attn_applied = attn_applied.permute(1,0,2)

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)

            use_teacher_forcing = True# if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                decoder_output_full[idx, :, :] = decoder_output
                decoder_input = target[idx-1]  # Teacher forcing

            else:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                # decoder_input_v = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                # print decoder_input
                decoder_output_full[idx, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        # gen_output = self.softmax(self.out(decoder_output_full))

        return decoder_output_full

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        SOS_token = autograd.Variable(SOS_token)
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

    def initHidden(self):
        result = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class GeneratorEncDecTeacherForcingNoAtt(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcingNoAtt, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim)#, n_layers, dropout=0.5, batch_first=True)

        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        motion = motion.permute(1,0)
        encoder_outputs = autograd.Variable(torch.zeros(self.motion_length, self.batch_size, self.hidden_dim))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        for idx, motion_word in enumerate(motion):
            if idx == 0:
                encoder_hidden = self.initHidden()
            embedded = self.encoder_embeddings(motion_word).view(1, self.batch_size, -1)
            encoder_output, encoder_hidden = self.encoder_gru(embedded, encoder_hidden) #[batch_size, seq_length, hidden_dim]
            encoder_outputs[idx] = encoder_output[0]
        return encoder_outputs, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.vocab_size))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        target = target_output.permute(1,0)

        for idx in range(self.claim_length):
            if idx == 0:
                decoder_input = SOS_token
                decoder_hidden = encoder_hidden
            output = self.decoder_embeddings(decoder_input).view(1, self.batch_size, -1)
            output = self.dropout(output)

            use_teacher_forcing = True# if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                decoder_output_full[idx, :, :] = decoder_output
                decoder_input = target[idx-1]  # Teacher forcing

            else:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                # decoder_input_v = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                # print decoder_input
                decoder_output_full[idx, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        # gen_output = self.softmax(self.out(decoder_output_full))

        return decoder_output_full

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        SOS_token = autograd.Variable(SOS_token)
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

    def initHidden(self):
        result = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class GeneratorEncDecTeacherForcingNoAttSelu(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcingNoAttSelu, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim)#, n_layers, dropout=0.5, batch_first=True)

        self.selu = nn.SELU()
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        motion = motion.permute(1,0)
        encoder_outputs = autograd.Variable(torch.zeros(self.motion_length, self.batch_size, self.hidden_dim))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        for idx, motion_word in enumerate(motion):
            if idx == 0:
                encoder_hidden = self.initHidden()
            embedded = self.encoder_embeddings(motion_word).view(1, self.batch_size, -1)
            encoder_output, encoder_hidden = self.encoder_gru(embedded, encoder_hidden) #[batch_size, seq_length, hidden_dim]
            encoder_outputs[idx] = encoder_output[0]
        return encoder_outputs, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.vocab_size))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        target = target_output.permute(1,0)

        for idx in range(self.claim_length):
            if idx == 0:
                decoder_input = SOS_token
                decoder_hidden = encoder_hidden
            output = self.decoder_embeddings(decoder_input).view(1, self.batch_size, -1)
            output = self.dropout(output)

            output = self.selu(output)

            use_teacher_forcing = True# if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                decoder_output_full[idx, :, :] = decoder_output
                decoder_input = target[idx-1]  # Teacher forcing

            else:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                # decoder_input_v = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                # print decoder_input
                decoder_output_full[idx, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        # gen_output = self.softmax(self.out(decoder_output_full))

        return decoder_output_full

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        SOS_token = autograd.Variable(SOS_token)
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

    def initHidden(self):
        result = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class GeneratorEncDecTeacherForcingNoAttSeluShared(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, encoder, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcingNoAttSeluShared, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size
        self.vocab_size = vocab_size

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_p)
        self.selu = nn.SELU()
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def decode(self, SOS_token, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.vocab_size))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        target = target_output.permute(1,0)

        for idx in range(self.claim_length):
            if idx == 0:
                decoder_input = SOS_token
                decoder_hidden = encoder_hidden.unsqueeze(0)
            output = self.decoder_embeddings(decoder_input).view(1, self.batch_size, -1)
            output = self.dropout(output)

            output = self.selu(output)

            use_teacher_forcing = True# if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                decoder_output_full[idx, :, :] = decoder_output
                decoder_input = target[idx-1]  # Teacher forcing

            else:
                decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
                decoder_output = F.log_softmax(self.out(decoder_output[0]), dim=1)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                # decoder_input_v = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = autograd.Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                # print decoder_input
                decoder_output_full[idx, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        # gen_output = self.softmax(self.out(decoder_output_full))

        return decoder_output_full

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_feat, _ = self.encoder(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        SOS_token = autograd.Variable(SOS_token)
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)

        gen_output = self.decode(SOS_token, encoder_feat, target_output, teacher_forcing_ratio)

        return gen_output

    def initHidden(self):
        result = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class GeneratorVan(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorVan, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim*2, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, motion):
        # ENCODER BLOCK
        # hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden

        embedded = self.embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        gen_output = self.softmax(self.out(encoder_output))

        return gen_output
