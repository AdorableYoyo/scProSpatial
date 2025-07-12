import re
import torch
from torch import nn
from performer_pytorch import *
from math import ceil
from utils import *
import pandas as pd

ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['mask'])
    return enc_kwargs, dec_kwargs, kwargs

#################################################
#-------------------- Model --------------------#
#################################################

class scPerformerLM(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        dim,depth,
        heads,
        num_tokens=1,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False,
        shift_tokens = False,
        gene_emb_file = None,
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.to_vector = nn.Linear(1,dim)
        self.pos_emb = nn.Embedding(22607,dim,padding_idx=0)# There are 22606 NCBI Gene ID obtained on mar 12th, 2025 in my dataset ( protein + rna combined)
    
        if gene_emb_file is not None:
            print('Loading gene embeddings from file {}'.format(gene_emb_file))
            gene_emb_df = pd.read_csv(gene_emb_file, index_col = 'MYID')
            for my_Id in gene_emb_df.index:
                self.pos_emb.weight.data[my_Id] = torch.tensor(gene_emb_df.loc[my_Id].values, dtype=torch.float32)
            #freeze the gene embeddings
        
            #self.pos_emb.weight.requires_grad = False
        self.layer_pos_emb = Always(None)
        self.dropout = nn.Dropout(emb_dropout)
        self.performer = Performer(dim, depth, heads, dim_head)
        # self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None
        self.dataset_emb = nn.Embedding(6,dim) # six
        self.alpha = nn.Parameter(torch.ones(1))
    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, geneID, data_id=None, labels=None, **kwargs):
        b, n = x.shape[0], x.shape[1]
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=2)
            x = self.to_vector(x)
        x = x + self.pos_emb(geneID)
        x = self.dropout(x)
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)
        if data_id is not None:
            dataset_emb = self.dataset_emb(data_id) # should be batch_size, dim
            # all the samples are from the same dataset  shape : dim
            x = x + self.alpha*dataset_emb.unsqueeze(1)
        if labels is not None:
            mask = labels > 0  # Assuming labels > 0 are masked
            x_out = torch.squeeze(self.to_out(x))  # batch, seq_len # torch.Size([8, 2000])
            x_out = x_out[mask]  # get predictions for masked positions # 4800 ? 
            #labels = labels[mask]
            return x, x_out
        else:
            x_out = torch.squeeze(self.to_out(x))
            return x, x_out



class MLPTranslator(nn.Module):
    """
    Class description: translator from RNA to protein
    fully connected layer with adjustable number of layers and variable dropout for each layer
    
    """
    #----- Define all layers -----#
    def __init__(self, num_fc_input, num_output_nodes, num_fc_layers, initial_dropout, act = nn.ReLU(), **kwargs):
        super(MLPTranslator, self).__init__(**kwargs)
        fc_d = pow(num_fc_input/num_output_nodes,1/num_fc_layers) # reduce factor of fc layer dimension
        #--- Fully connected layers ---#
        self.num_fc_layers = num_fc_layers
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input/fc_d)))
            self.dropout0 = nn.Dropout(initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input/fc_d)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1,num_fc_layers-1):
                    tmp_input = int(ceil(num_fc_input/fc_d**i))
                    tmp_output = int(ceil(num_fc_input/fc_d**(i+1)))
                    exec('self.fc{} = nn.Linear(tmp_input, tmp_output)'.format(i))
                    if i < ceil(num_fc_layers/2) and 1.1**(i+1)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)'.format(i))
                    elif i >= ceil(num_fc_layers/2) and 1.1**(num_fc_layers-1-i)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)'.format(i))
                    else:
                        exec('self.dropout{} = nn.Dropout(initial_dropout)'.format(i))
                # the last fc layer
                exec('self.fc{} = nn.Linear(tmp_output, num_output_nodes)'.format(i+1))
            
        #--- Activation function ---#
        self.act = act
    
    #----- Forward -----#
    def forward(self, x):
        # x size:  [batch size, feature_dim] 
        
        if self.num_fc_layers == 1:
            outputs = self.fc0(x)
        else:
            # the first fc layer
            outputs = self.act(self.dropout0(self.fc0(x)))
            if self.num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                outputs = self.fc1(outputs)
            else:
                # the middle fc layer
                for i in range(1,self.num_fc_layers-1):
                    outputs = eval('self.act(self.dropout{}(self.fc{}(outputs)))'.format(i,i))
                # the last fc layer
                outputs = eval('self.fc{}(outputs)'.format(i+1))
            
        return outputs


class RNA_pretrain(nn.Module):
    def __init__(
        self,
        dim,
        no_projection = False,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dim
        enc_kwargs['no_projection'] = no_projection
        self.rna_enc = scPerformerLM(**enc_kwargs)
        self.rna_proj = nn.Linear(dim, dim//2)
        # self.dataset_emb = nn.Embedding(6,dim) # six
        # self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, rna_id, rna_x, data_id = None, rna_mask_label=None, get_emb = None, solo_emb = None, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        if get_emb :
            rna_emb, _ = self.rna_enc(rna_x, rna_id,data_id, rna_mask_label, **enc_kwargs)
            if solo_emb:
                return rna_emb.mean(dim=1)
            rna_joined = self.rna_proj(rna_emb)
            return rna_joined.mean(dim=1)
        else:
            _, rna_out= self.rna_enc(rna_x, rna_id, data_id, rna_mask_label, **enc_kwargs) 
          
            
            return rna_out
        
class Prot_pretrain(nn.Module):
    def __init__(
        self,
        dim,
        no_projection = False,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        dec_kwargs['dim'] = dim
        dec_kwargs['no_projection'] = no_projection
        self.prot_enc = scPerformerLM(**dec_kwargs)
        self.prot_proj = nn.Linear(dim, dim//2)
    
    def forward(self, prot_id,  prot_x, data_id=None,prot_mask_label=None, get_emb = None, solo_emb = None, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        if get_emb :
            prot_emb, _ = self.prot_enc(prot_x, prot_id, data_id, prot_mask_label, **dec_kwargs)
            if solo_emb:
                return prot_emb.mean(dim=1)
            protein_joined = self.prot_proj(prot_emb)
            return protein_joined.mean(dim=1)
        else:
            _, prot_out= self.prot_enc(prot_x, prot_id,data_id, prot_mask_label, **dec_kwargs) 
            return prot_out
        

class Encoders(nn.Module):
    def __init__(
        self,
        dim,
        no_projection = False,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['no_projection'] = dec_kwargs['no_projection'] = no_projection
        self.rna_enc = scPerformerLM(**enc_kwargs)
        self.prot_enc = scPerformerLM(**dec_kwargs)
        self.rna_proj = nn.Linear(dim, dim//2)
        self.prot_proj = nn.Linear(dim, dim//2)

    def forward(self, rna_id, rna_x, prot_id, prot_x,rna_mask_label=None, prot_mask_label=None,get_emb = None, solo_emb = None, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        if get_emb :
            rna_emb, _ = self.rna_enc(rna_x, rna_id, rna_mask_label, **enc_kwargs)
            prot_emb, _ = self.prot_enc(prot_x, prot_id, prot_mask_label, **dec_kwargs)
            if solo_emb:
                return rna_emb.mean(dim=1), prot_emb.mean(dim=1)
            # pooled_rna = rna_emb.mean(dim=1)
            # pooled_prot = prot_emb.mean(dim=1)
            rna_joined = self.rna_proj(rna_emb)
            protein_joined = self.prot_proj(prot_emb)
            return rna_joined.mean(dim=1), protein_joined.mean(dim=1)
            

            #return rna_emb, prot_emb
            #return pooled_rna, pooled_prot
        else:
            _, rna_out= self.rna_enc(rna_x, rna_id, rna_mask_label, **enc_kwargs) 
            # expected dim : batch_size, 2000
            _, prot_out= self.prot_enc(prot_x, prot_id, prot_mask_label, **dec_kwargs)
            return rna_out, prot_out

        
class CellType_Pred(Encoders):
    def __init__(
        self, 
        dim,
        initial_dropout,
        num_fc_input, num_output_nodes, num_fc_layers=2,
        act = nn.ReLU(),**kwargs
    ):
        super().__init__(dim, **kwargs)
   
        fc_d = pow(num_fc_input/num_output_nodes,1/num_fc_layers) # reduce factor of fc layer dimension
        self.num_fc_layers = num_fc_layers
        self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input/fc_d)))
        self.dropout0 = nn.Dropout(initial_dropout)
        self.fc1 = nn.Linear(int(ceil(num_fc_input/fc_d)), num_output_nodes)
        self.act = act
        #self.freeze_joint_embedding()
   
    def freeze_joint_embedding(self):
        """
        Freeze all parameters up to and including the prot_proj layer.
        Only fc0 and fc1 will have trainable parameters.
        """
        for name, param in self.named_parameters():
            # Check if the parameter belongs to layers before or including prot_proj
            if "proj" not in name and not name.startswith("fc"):
                param.requires_grad = False
                print(f"Freezing parameter: {name}")
                
    def forward(self, input_type, id_data, x_data, **kwargs):
        """
        Switch between RNA or protein forward based on the input_type.
        
        Args:
            input_type (str): Should be 'rna' for RNA data or 'prot' for protein data.
            id_data (Tensor): Input identifier data for RNA or protein.
            x_data (Tensor): Input feature data for RNA or protein.
            kwargs: Additional arguments for encoders.
        
        Returns:
            Tensor: Processed outputs based on the selected input type.
        """
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        
        if input_type == 'prot':
            # Protein pathway
            prot_emb, _ = self.prot_enc(x_data, id_data, **dec_kwargs)
            joined_emb = self.prot_proj(prot_emb)  # Project to joint space
        elif input_type == 'rna':
            # RNA pathway
            rna_emb, _ = self.rna_enc(x_data, id_data, **enc_kwargs)
            joined_emb = self.rna_proj(rna_emb)  # Project to joint space (using same projection layer)
   
        else:
            raise ValueError("input_type must be either 'rna' or 'prot'")
  
        # Add MLP for cell type prediction
        outputs = self.act(self.dropout0(self.fc0(joined_emb)))
        outputs = self.fc1(outputs)
        return outputs.mean(dim=1)
    
class Zero_CellType_Pred(Encoders):
    def __init__(
        self, 
        dim,
        initial_dropout,
        num_fc_input, num_output_nodes, num_fc_layers=2,
        act = nn.ReLU(),**kwargs
    ):
        super().__init__(dim, **kwargs)
   

        self.fc1 = nn.Linear(num_fc_input*2, num_output_nodes)
        
    def forward(self, input_type, id_data, x_data, **kwargs):
        """
        Switch between RNA or protein forward based on the input_type.
        
        Args:
            input_type (str): Should be 'rna' for RNA data or 'prot' for protein data.
            id_data (Tensor): Input identifier data for RNA or protein.
            x_data (Tensor): Input feature data for RNA or protein.
            kwargs: Additional arguments for encoders.
        
        Returns:
            Tensor: Processed outputs based on the selected input type.
        """
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        
        if input_type == 'prot':
            # Protein pathway
            prot_emb, _ = self.prot_enc(x_data, id_data, **dec_kwargs)
            outputs = self.fc1(prot_emb)
           
        elif input_type == 'rna':
            # RNA pathway
            rna_emb, _ = self.rna_enc(x_data, id_data, **enc_kwargs)
            outputs = self.fc1(rna_emb)

        else:
            raise ValueError("input_type must be either 'rna' or 'prot'")

        return outputs.mean(dim=1)
    
class RNA2Prot(nn.Module):
    def __init__(
        self, 
        dim,
        translator_depth, initial_dropout,**kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
      
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        self.rna_enc = scPerformerLM(**enc_kwargs)
        self.prot_enc = scPerformerLM(**dec_kwargs)
        self.rna_proj = nn.Linear(dim, dim//2)
        self.prot_proj = nn.Linear(dim, dim//2)

        self.fc0 = nn.Linear(dim//2,dim )
        self.dropout0 = nn.Dropout(initial_dropout)
        self.translator = MLPTranslator(enc_kwargs['max_seq_len'], dec_kwargs['max_seq_len'], translator_depth, initial_dropout)

    def load_pretrained(self, rna_model_path, prot_model_path):
        rna_state_dict = torch.load(rna_model_path)  # Load as dictionary
        prot_state_dict = torch.load(prot_model_path)
        # Load weights correctly into initialized model
  
        # Load the full state dictionary into the RNA encoder and projector
        self.rna_enc.load_state_dict({k.replace("rna_enc.", ""): v for k, v in rna_state_dict.items() if k.startswith("rna_enc")})
        self.rna_proj.load_state_dict({k.replace("rna_proj.", ""): v for k, v in rna_state_dict.items() if k.startswith("rna_proj")})

        # Load the full state dictionary into the protein encoder and projector
        self.prot_enc.load_state_dict({k.replace("prot_enc.", ""): v for k, v in prot_state_dict.items() if k.startswith("prot_enc")})
        self.prot_proj.load_state_dict({k.replace("prot_proj.", ""): v for k, v in prot_state_dict.items() if k.startswith("prot_proj")})


        print("Pretrained models loaded successfully from files {} and {}".format(rna_model_path, prot_model_path))
    
    def forward(self, rna_id, rna_x, prot_id, data_id = None,**kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        rna_emb, _ = self.rna_enc(rna_x, rna_id,data_id, **enc_kwargs)
        # rna_joined = self.rna_proj(rna_emb) #  embeddings from the joint space
       
        # outputs = self.dropout0(self.fc0(rna_joined)) # to the protein space
        seq_out = self.translator(rna_emb.transpose(1, 2).contiguous()).transpose(1, 2).contiguous().clone()
        protein_emb, protein_predictions= self.prot_enc(seq_out, prot_id, data_id, **dec_kwargs)

        
        return  protein_predictions