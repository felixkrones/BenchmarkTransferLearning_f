import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def get_prepared_checkpoint(model, checkpoint_path):
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    for k_name in checkpoint.keys():
        if 'model' in k_name or k_name=='teacher':
            nm = k_name
            checkpoint_model = checkpoint[nm]
            break
        else: 
            checkpoint_model = checkpoint

    checkpoint_model = {k.replace("img_encoder_q.model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        # if k in checkpoint_model:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    
    return model


class LabelTokenViT(nn.Module):

    def __init__(self, args, model, label_layers=4, label_cross=False, num_tokens=None, norm_lt=True):
        super(LabelTokenViT, self).__init__()

        self.vit = model
        self.embed_dim = model.embed_dim
        self.nb_classes = args.nb_classes
        self.depth = len(self.vit.blocks)

        if label_layers > 0:
            self.useLT = True
            self.label_layers = [self.depth-label_layers+i for i in range(label_layers)]
            self.num_tokens = self.nb_classes if num_tokens is None else num_tokens

            self.label_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
            self.label_tokens = trunc_normal_(self.label_tokens, std=0.2)
            
            self.label_heads = nn.ModuleList([nn.Linear(self.embed_dim, 1) for i in range(self.nb_classes)])
            self.norm_lt = True if args.norm_lt==1 else False

        else:
            self.useLT = False
            self.label_layers = [0]

    def add_label_tokens(self, x):
        # [class token, label tokens, data tokens]

        B = x.shape[0]
        label_tokens = self.label_tokens.expand(B, -1, -1)
        x = torch.cat((x[:,0].unsqueeze(1), label_tokens, x[:,1:]), dim=1)

        return x

    def interp_label_tokens(self, x):

        return x

    def compute_labels(self, x):

        num_tokens = self.num_tokens
        n_class = self.nb_classes

        x = x[:, 1:num_tokens+1, :]

        # TODO: write code to extrapolate to label dimension
        x = self.interp_label_tokens(x)

        if self.norm_lt:
            x = torch.cat([self.label_heads[i](self.vit.norm(x[:,i,:])) for i in range(n_class)], dim=-1)
        else:
            x = torch.cat([self.label_heads[i](x[:,i,:]) for i in range(n_class)], dim=-1)

        return x

    def forward(self, x, class_blocks='', classify=False):

        c_blcks = [-1] if class_blocks=='' else list(map(int, class_blocks.split('-')))

        x = self.vit.prepare_tokens(x)

        x_agg, cnt = 0, 0
        for layer_idx, blk in enumerate(self.vit.blocks):

            if layer_idx == self.label_layers[0] and self.useLT:
                x = self.add_label_tokens(x)

            x = blk(x)

            if (layer_idx+1) in c_blcks:
                #TODO: need to add the norm layer from the next block
                if (layer_idx+1) != len(self.vit.blocks): 
                    x_agg = x_agg + self.vit.blocks[layer_idx+1].norm1(x)
                else:
                    x_agg = x_agg + self.vit.norm(x)
                cnt += 1

        if cnt != 0:
            x_agg /= cnt # getting the average
            # recons = self.vit.norm(recons)
            
        x = self.vit.norm(x)

        if classify:
            x = x if class_blocks == '' else x_agg
            out = self.compute_labels(x) if self.useLT else self.vit.head_class(x[:, 0])
        elif self.useLT:  
            out = x[:, 0], x[:, 1:self.num_tokens + 1], x[:, self.num_tokens + 1:]
        else:
            out = x[:, 0], x[:, 1:]


        return out


    def get_all_attention(self, x):
        x = self.vit.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []

        for layer_idx, blk in enumerate(self.vit.blocks):

            # print((self.label_layers != 0) and (layer_idx == self.label_layers[0]))

            if layer_idx == self.label_layers[0] and self.useLT:
                x = self.add_label_tokens(x)

            x, x_att = blk(x, return_attention=True)

            output.append(x_att)

        x = self.vit.norm(x)

        return output
