# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
The CumSoftmax function in paper 
cummax(x) = cumsum (softmax(x))
"""

def cummax(x):
    return torch.cumsum(F.softmax(x, -1), dim=-1)


# define leaner layer with dropout
class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super().__init__(in_features, out_features, bias)
        self.dropout = dropout

    def _create_mask(self):
        """
        Sample a dropout mask if dropout rate is set.
        """
        if self.dropout == 0.:
            return self.weight
        mask = torch.bernoulli(self.dropout * torch.ones_like(self.weight)).bool()
        return self.weight.masked_fill(mask, 0.)

    def forward(self, input, use_mask=False):
        """
        Compute the forward pass. If training, apply dropout to the weights.
        """
        if self.training and use_mask:
            weight = self._create_mask()
        else:
            weight = self.weight * (1 - self.dropout)
        
        return F.linear(input, weight, self.bias)



class PFN_Unit(nn.Module):
    def __init__(self, args, input_size):
        super(PFN_Unit, self).__init__()
        self.args = args

        self.hidden_transform = LinearDropConnect(args.hidden_size, 5 * args.hidden_size, bias=True, dropout=args.dropconnect)
        self.input_transform = nn.Linear(input_size, 5 * args.hidden_size, bias=True)
        self.transform = nn.Linear(args.hidden_size * 3, args.hidden_size)

        # Modules using DropConnect
        self.drop_weight_modules = [self.hidden_transform]

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m._create_mask()

    def forward(self, x, hidden):
        h_in, c_in = hidden

        # Calculate gates
        gates = self.input_transform(x) + self.hidden_transform(h_in)
        c, eg_cin, rg_cin, eg_c, rg_c = gates.chunk(5, 1)

        eg_cin, rg_cin = 1 - cummax(eg_cin), cummax(rg_cin)
        eg_c, rg_c = 1 - cummax(eg_c), cummax(rg_c)
        c = torch.tanh(c)

        # Compute overlaps for 'c' and 'c_in'
        overlap_c, upper_c, downer_c = rg_c * eg_c, rg_c - rg_c * eg_c, eg_c - rg_c * eg_c
        overlap_cin, upper_cin, downer_cin = rg_cin * eg_cin, rg_cin - rg_cin * eg_cin, eg_cin - rg_cin * eg_cin

        share = overlap_cin * c_in + overlap_c * c

        # Compute c_re, c_ner, c_share
        c_re = upper_cin * c_in + upper_c * c + share
        c_ner = downer_cin * c_in + downer_c * c + share
        c_share = share

        # tanh transform for all gates
        h_re, h_ner, h_share = torch.tanh(c_re), torch.tanh(c_ner), torch.tanh(c_share)

        c_out = self.transform(torch.cat((c_re, c_ner, c_share), dim=-1))
        h_out = torch.tanh(c_out)

        return (h_out, c_out), (h_ner, h_re, h_share)
    

class NERUnit(nn.Module):
    def __init__(self, args, ner2idx):
        super(NERUnit, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.ner2idx = ner2idx

        self.init_layers()
        
    def init_layers(self):
        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2tag = nn.Linear(self.hidden_size, len(self.ner2idx))
        self.elu = nn.ELU()
        self.n = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, h_ner, h_share, mask):
        length, batch_size, _ = h_ner.size()
        h_global = self.calculate_global_feature(h_ner, h_share)
        ner = self.construct_ner_feature(h_ner, h_global)
        ner = self.process_ner_feature(ner)
        ner = self.apply_mask(length, batch_size, ner, mask)

        return ner

    def calculate_global_feature(self, h_ner, h_share):
        h_global = torch.cat((h_share, h_ner), dim=-1)
        h_global = torch.tanh(self.n(h_global))
        h_global = torch.max(h_global, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(h_ner.size(0), 1, 1)
        h_global = h_global.unsqueeze(0).repeat(h_ner.size(0), 1, 1, 1)
        return h_global

    def construct_ner_feature(self, h_ner, h_global):
        length, batch_size, _ = h_ner.size()
        st = h_ner.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner.unsqueeze(0).repeat(length, 1, 1, 1)
        ner = torch.cat((st, en, h_global), dim=-1)
        return ner

    def process_ner_feature(self, ner):
        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))
        return ner

    def apply_mask(self,length, batch_size, ner, mask):
        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(device)
        diagonal_mask = diagonal_mask.permute(1, 2, 0)
        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)
        mask_ner = mask_s * mask_e
        mask = diagonal_mask * mask_ner
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, len(self.ner2idx))
        ner = ner * mask
        return ner



class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()
        self.args = args
        self.unit = PFN_Unit(args, input_size)

    def hidden_init(self, batch_size):
        h0 = torch.zeros(batch_size, self.args.hidden_size).requires_grad_(False).to(device)
        c0 = torch.zeros(batch_size, self.args.hidden_size).requires_grad_(False).to(device)
        return (h0, c0)

    def forward(self, x):
        seq_len = x.size(0)
        batch_size = x.size(1)
        h_ner, h_re, h_share = self.process_sequence(x, seq_len, batch_size)
        return h_ner, h_re, h_share

    def process_sequence(self, x, seq_len, batch_size):
        h_ner, h_re, h_share = [], [], []
        hidden = self.hidden_init(batch_size)

        if self.training:
            self.unit.sample_masks()

        for t in range(seq_len):
            hidden, h_task = self.unit(x[t, :, :], hidden)
            h_ner.append(h_task[0])
            h_re.append(h_task[1])
            h_share.append(h_task[2])

        h_ner = torch.stack(h_ner, dim=0)
        h_re = torch.stack(h_re, dim=0)
        h_share = torch.stack(h_share, dim=0)

        return h_ner, h_re, h_share

class ReUnit(nn.Module):
    def __init__(self, args, re2idx):
        super(ReUnit, self).__init__()
        self.hidden_size = args.hidden_size
        self.relation_size = len(re2idx)
        self.re2idx = re2idx

        self.init_layers(args)
        
    def init_layers(self,args):
        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, self.relation_size)
        self.elu = nn.ELU()
        self.r = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h_re, h_share, mask):
        length, batch_size, _ = h_re.size()
        h_global = self.calculate_global_feature(length, h_re, h_share)
        re = self.construct_relation_feature(h_re, h_global)
        re = self.process_relation_feature(re)
        re = self.apply_mask(length, re, mask)

        return re

    def calculate_global_feature(self, length, h_re, h_share):
        h_global = torch.cat((h_share, h_re), dim=-1)
        h_global = torch.tanh(self.r(h_global))
        h_global = torch.max(h_global, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)
        return h_global

    def construct_relation_feature(self, h_re, h_global):
        length, batch_size, _ = h_re.size()
        r1 = h_re.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re.unsqueeze(0).repeat(length, 1, 1, 1)
        re = torch.cat((r1, r2, h_global), dim=-1)
        return re

    def process_relation_feature(self, re):
        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))
        return re

    def apply_mask(self, length, re, mask):
        mask = mask.unsqueeze(-1).repeat(1, 1, self.relation_size)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask = mask_e1 * mask_e2
        re = re * mask
        return re
    
    

class PFN(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(PFN, self).__init__()
        self.args = args
        self.feature_extractor = self.create_encoder(args, input_size)
        self.ner = NERUnit(args, ner2idx)
        self.re = ReUnit(args, rel2idx)
        self.dropout = nn.Dropout(args.dropout)
        self.tokenizer, self.bert = self.load_embed_model(args)

    def create_encoder(self, args, input_size):
        return Encoder(args, input_size)

    def load_embed_model(self, args):
        if args.embed_mode == 'albert':
            tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            bert = AlbertModel.from_pretrained("albert-xxlarge-v1")
        elif args.embed_mode == 'bert_cased':
            tokenizer = AutoTokenizer.from_pretrained("./bert")
            bert = AutoModel.from_pretrained("./bert")
        elif args.embed_mode == 'scibert':
            tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        return tokenizer, bert

    def forward(self, x, mask):
        x = self.tokenize_and_encode(x)
        if self.training:
            x = self.dropout(x)
        h_ner, h_re, h_share = self.feature_extractor(x)
        ner_score = self.ner(h_ner, h_share, mask)
        re_core = self.re(h_re, h_share, mask)
        return ner_score, re_core

    def tokenize_and_encode(self, x):
        x = self.tokenizer(x, return_tensors="pt", padding='longest', is_split_into_words=True).to(device)
        x = self.bert(**x)[0]
        x = x.transpose(0, 1)
        return x
