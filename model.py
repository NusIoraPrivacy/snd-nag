import torch
from torch import nn
import torch.nn.functional as F
import random
# from dynamic_crf_layer import *
from dynamic_crf_layer import *
from torch.distributions.gamma import Gamma

def sample_noise_Gaussian(d_shape, noise_stddev, device="cpu"):
    noise = torch.normal(mean=0., std=noise_stddev, size=d_shape, device=device)
    return noise

def sample_noise_Chi(d_shape, eta, device="cpu"):
    n_dim = d_shape[-1]
    alpha = torch.ones(d_shape) * n_dim
    # eta:卡方分布参数
    beta = torch.ones(d_shape) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    v_lst = -2 * torch.rand(d_shape) + 1
    noise = l_lst * v_lst
    noise = noise.to(device)
    return noise

# NAG-BERT的decoder
class TopLayer(nn.Module):
    # crf_low_rank是低秩矩阵的秩
    def __init__(self, vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, padding_idx):
        super(TopLayer, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.crf_layer = DynamicCRF(num_embedding = vocab_size, low_rank = crf_low_rank, 
                                    beam_size = crf_beam_size)

        self.one_more_layer_norm = nn.LayerNorm(embed_dim)
        self.tgt_word_prj = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, src_representation, src_input, tgt_input, is_training):
        '''
            src_representation : bsz x seqlen x embed_dim
            src_input : bsz x seqlen
            tgt_input : bsz x seqlen
        '''
        assert src_input.size() == tgt_input.size()

        src_input = src_input.transpose(0, 1) # src_len x bsz
        seqlen, bsz = src_input.size()

        src_representation = F.dropout(src_representation, p=self.dropout, training=is_training)
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim

        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)
        log_probs = torch.log_softmax(emissions, -1)
        assert log_probs.size() == torch.Size([seqlen, bsz, self.vocab_size])

        emissions = emissions.transpose(0, 1) # [bsz x src_len x vocab_size]
        emission_mask = ~tgt_input.eq(self.padding_idx) # [bsz x src_len]
        batch_crf_loss = -1 * self.crf_layer(emissions, tgt_input, emission_mask) # [bsz]
        assert batch_crf_loss.size() == torch.Size([bsz])
        return log_probs, batch_crf_loss

    def decoding(self, src_representation, src_input):
        '''
            src_representation : bsz x seqlen x embed_dim
            src_input : bsz x seqlen
            tgt_input : bsz x seqlen
        '''
        src_input = src_input.transpose(0, 1) # src_len x bsz
        seqlen, bsz = src_input.size()

        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        _, finalized_tokens = self.crf_layer.forward_decoder(emissions)
        assert finalized_tokens.size() == torch.Size([bsz, seqlen])
        return finalized_tokens

    def length_ratio_decoding(self, src_representation, src_input, length_ratio):
        '''
            src_representation : 1 x seqlen x embed_dim
            src_input : 1 x seqlen
        '''
        src_input = src_input.transpose(0, 1) # src_len x bsz
        seqlen, bsz = src_input.size()

        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        valid_len = int(seqlen * length_ratio) + 1
        valid_emissions = emissions[:, :valid_len+1,:]
        _, finalized_tokens = self.crf_layer.forward_decoder(valid_emissions)
        return finalized_tokens

class NAG_BERT(nn.Module):
    def __init__(self, bert_model, vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, src_padding_idx, tgt_padding_idx):
        super(NAG_BERT, self).__init__()
        self.embed_dim = embed_dim
        self.bert_model = bert_model
        self.toplayer = TopLayer(vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, tgt_padding_idx)
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        # self.attention_mask = None

    def forward(self, src_input, tgt_input, is_training):
        '''
            src_input : bsz x seqlen
            tgt_input : bsz x seqlen 
        '''
        bsz, seqlen = src_input.size()
        # src_mask:真实值 = attention_mask
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model(src_input, attention_mask = src_mask,
                                                output_all_encoded_layers = False)
        assert src_representation.size() == torch.Size([bsz, seqlen, self.embed_dim])
        # 获得概率和crf损失，输出非自回归的文本摘要
        log_probs, batch_crf_loss = self.toplayer(src_representation, src_input, tgt_input, is_training)
        return log_probs, batch_crf_loss

    # 获取encoder输出
    def get_src_representation(self, src_input):
        # bsz, seqlen = src_input.size()
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model(src_input, attention_mask=src_mask, output_all_encoded_layers=False)
        return src_representation

    # 用于验证集dev
    def decoding(self, src_input):
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model.work(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)
        finalized_tokens = self.toplayer.decoding(src_representation, src_input)
        return finalized_tokens

    # 这个只在inference.py里引用了，另一个length_ratio_decoding在toplayer类里
    def length_ratio_decoding(self, src_input, length_ratio):
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)

        finalized_tokens = self.toplayer.length_ratio_decoding(src_representation, 
                            src_input, length_ratio)
        return finalized_tokens

    def length_ratio_decoding_no_dropout(self, src_input, length_ratio):
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model.work(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)

        finalized_tokens = self.toplayer.length_ratio_decoding(src_representation, 
                            src_input, length_ratio)
        return finalized_tokens

    # local token representation
    def get_local_embedding(self, src_input, args):
        src_input = torch.tensor(src_input).to(args.gpu_id)
        init_embs = self.bert_model.embeddings.word_embeddings(src_input)
        return init_embs

    # privatization module
    def get_noisy_embedding(self, src_input, args, mode="train"):
        init_embs = self.get_local_embedding(src_input, args)
        # sample noise
        if args.noise_mechanism == "Gaussian":
            noise_std = args.train_noise_std if mode == "train" else args.test_noise_std
            noises = sample_noise_Gaussian(init_embs.shape, noise_std, args.gpu_id)
        elif args.noise_mechanism == "ChiDP":
            eta = args.train_eta if mode == "train" else args.test_eta
            noises = sample_noise_Chi(init_embs.shape, eta, args.gpu_id)
        noise_init_emb = init_embs + noises
        return noise_init_emb, noises

    # server encoder&decoder
    def get_server_outputs(self, args, init_embs, noise_init_emb, attention_mask, decoder_start_token=None):
        # ======= encoder =======
        # obtain final embeddings without noise
        with torch.no_grad():
            if "t5" in args.base_model:
                decoder_start_token = decoder_start_token.repeat(init_embs.shape[0], 1)
                encoder_outputs = self.bert_model(inputs_embeds=init_embs, attention_mask=attention_mask,
                                decoder_input_ids=decoder_start_token, output_hidden_states=True)
            else:
                encoder_outputs = self.bert_model(inputs_embeds=init_embs, attention_mask=attention_mask, output_hidden_states=True)

        # get final embeddings with noise
        with torch.no_grad():
            if "t5" in args.base_model:
                encoder_noise_outputs = self.bert_model(inputs_embeds=noise_init_emb, attention_mask=attention_mask,
                                      decoder_input_ids=decoder_start_token, output_hidden_states=True)
            else:
                encoder_noise_outputs = self.bert_model(inputs_embeds=noise_init_emb, attention_mask=attention_mask,
                                      output_hidden_states=True)
        # ======= decoder =======
        # self.toplayer.





