import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):

    def __init__(self, q_input_dim, cand_input_dim, v_dim, kq_dim=64):
        super().__init__()
        
        self.query_transform = nn.Linear(q_input_dim, kq_dim)
        self.key_transform = nn.Linear(cand_input_dim, kq_dim)
        self.value_transform = nn.Linear(cand_input_dim, v_dim)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.scale = math.sqrt(kq_dim)


    def forward(self, hidden, encoder_outputs):
        
        # q: B x kq_dim
        # k: B x T x kq_dim
        # v: B x T x v_dim
        
        q = self.query_transform(hidden)
        k = self.key_transform(encoder_outputs)
        v = self.value_transform(encoder_outputs)
        
        q = q.unsqueeze(1)      # B x 1 x kq_dim
        k = torch.permute(k, (0, 2, 1))     # B x kq_dim x T
    
    
        # B x 1 x T
        dot = torch.bmm(q, k)  
        scaled_dot = dot / self.scale
        
        # B x T
        scaled_dot = scaled_dot.squeeze(1)      
        alpha = self.softmax(scaled_dot)
        
        attended_val = torch.bmm(alpha.unsqueeze(1), v)
        attended_val = attended_val.squeeze(1)
        

        return attended_val, alpha



class Dummy(nn.Module):

    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim
        
    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], self.v_dim) ).to(hidden.device)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device)
        return zout, zatt

class MeanPool(nn.Module):

    def __init__(self, cand_input_dim, v_dim):
        super().__init__()
        self.linear = nn.Linear(cand_input_dim, v_dim)

    def forward(self, hidden, encoder_outputs):

        encoder_outputs = self.linear(encoder_outputs)
        output = torch.mean(encoder_outputs, dim=1)
        alpha = F.softmax(torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device), dim=-1)

        return output, alpha

class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab_len, emb_dim, enc_hid_dim, dropout=0.5):
        super().__init__()

        self.hidden_dim = enc_hid_dim

        self.embedding = nn.Embedding(src_vocab_len, emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)


    def forward(self, src, src_lens):
        
        embeddings = self.embedding(src)
        survivors = self.dropout(embeddings)
        word_representations, _ = self.gru(survivors)
        
        forward_states = word_representations[:, :, :self.hidden_dim]
        backward_states = word_representations[:, :, self.hidden_dim:]
        
        
        batch_idxs = torch.arange(forward_states.size(0), device=forward_states.device)
        time_idxs = src_lens - 1
        forward_final = forward_states[batch_idxs, time_idxs]
        
        backward_final = backward_states[:, 0, :]
        
        sentence_rep = torch.cat((forward_final, backward_final), dim=1)

        return word_representations, sentence_rep


class Decoder(nn.Module):
    def __init__(self, trg_vocab_len, emb_dim, dec_hid_dim, attention, dropout=0.5):
        super().__init__()

        self.attention = attention
        
        self.embedding = nn.Embedding(trg_vocab_len, emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(emb_dim, dec_hid_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(dec_hid_dim, dec_hid_dim),
            nn.GELU(),
            nn.Linear(dec_hid_dim, trg_vocab_len)
        )

    def forward(self, input, hidden, encoder_outputs):
    
        embeddings = self.embedding(input)
        survivors = self.dropout(embeddings)
        
        survivors = survivors.unsqueeze(1)
        hidden = hidden.unsqueeze(0)
        
        _, hidden = self.gru(survivors, hidden)
        
        hidden = hidden.squeeze(0)
        
        attended_feature, alphas = self.attention(hidden, encoder_outputs)
        hidden = hidden + attended_feature
        out = self.classifier(hidden)
        
        return hidden, out, alphas

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, kq_dim, attention, dropout=0.5):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = BidirectionalEncoder(src_vocab_size, embed_dim, enc_hidden_dim, dropout=dropout)
        self.enc2dec = nn.Sequential(nn.Linear(enc_hidden_dim*2, dec_hidden_dim), nn.GELU())

        if attention == "none":
            attn_model = Dummy(dec_hidden_dim)
        elif attention == "mean":
            attn_model = MeanPool(2*enc_hidden_dim, dec_hidden_dim)
        elif attention == "dotproduct":
            attn_model = DotProductAttention(dec_hidden_dim, 2*enc_hidden_dim, dec_hidden_dim, kq_dim)

        
        self.decoder = Decoder(trg_vocab_size, embed_dim, dec_hidden_dim, attn_model, dropout=dropout)
        



    def translate(self, src, src_lens, sos_id=1, max_len=50):
        
        #tensor to store decoder outputs and attention matrices
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)
        attns = torch.zeros(src.shape[0], max_len, src.shape[1]).to(src.device)

        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device)*sos_id

        with torch.no_grad():
            word_representations, sentence_rep = self.encoder(src, src_lens)
            hidden = self.enc2dec(sentence_rep)
            
            for t in range(max_len):
                hidden, out, alpha = self.decoder(input_words, hidden, word_representations)
                pred_words = out.argmax(dim=1)
                outputs[:, t] = pred_words
                attns[:, t, :] = alpha
                input_words = pred_words
            

        return outputs, attns
        

    def forward(self, src, trg, src_lens):

        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.trg_vocab_size).to(src.device)

        word_representations, sentence_rep = self.encoder(src, src_lens)
        hidden = self.enc2dec(sentence_rep)
        
        for t in range(1, trg.size(1)):
            hidden, out, _ = self.decoder(trg[:, t-1], hidden, word_representations)
            outputs[:, t, :] = out

        return outputs