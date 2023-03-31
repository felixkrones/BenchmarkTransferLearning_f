import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from functools import partial
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
import torch
import torch.nn as nn
import torchvision.models as models

import resnet_wider
import densenet
import simmim
from vits import VisionTransformerMoCo
from gmml.model_utils import get_prepared_checkpoint, LabelTokenViT
import gmml.data_transformations


def build_classification_model(args):
    if "vit" in args.model_name.lower():
        model = None
        if args.proxy_dir is None or args.proxy_dir =='':
            print('Loading pretrained {} weights for {} from timm.'.format(args.init, args.model_name))
            if args.model_name.lower() == "vit_base":
                if args.init.lower() =="random":
                    model = VisionTransformer(num_classes=args.num_class,
                            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    model.default_cfg = _cfg()
                    # model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=False)
                elif args.init.lower() =="imagenet_1k" or args.init.lower() =="imagenet":
                    model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=True)
                elif args.init.lower() =="imagenet_21k":
                    model = timm.create_model('vit_base_patch16_224_in21k', num_classes=args.num_class, pretrained=True)  
                elif args.init.lower() =="sam":
                    model = timm.create_model('vit_base_patch16_224_sam', num_classes=args.num_class, pretrained=True)
                elif args.init.lower() =="dino":
                    model = VisionTransformer(num_classes=args.num_class,
                            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    model.default_cfg = _cfg()
                    #model = timm.create_model('vit_base_patch16_224_dino', num_classes=args.num_class, pretrained=True) #not available in current timm version
                    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                    state_dict = torch.hub.load_state_dict_from_url(url=url)
                    model.load_state_dict(state_dict, strict=False)
                elif args.init.lower() =="deit":
                    model = timm.create_model('deit_base_patch16_224', num_classes=args.num_class, pretrained=True)
                elif args.init.lower() =="beit":
                    model = timm.create_model('beit_base_patch16_224', num_classes=args.num_class, pretrained=True)

            elif args.model_name.lower() == "vit_small":
                if args.init.lower() =="random":
                    model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=False)
                elif args.init.lower() =="imagenet_1k" or args.init.lower() =="imagenet":
                    model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=True)
                elif args.init.lower() =="imagenet_21k":
                    model = timm.create_model('vit_small_patch16_224_in21k', num_classes=args.num_class, pretrained=True)
                elif args.init.lower() =="dino":
                    #model = timm.create_model('vit_small_patch16_224_dino', num_classes=args.num_class, pretrained=True)
                    model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    model.default_cfg = _cfg()
                    url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                    state_dict = torch.hub.load_state_dict_from_url(url=url)
                    model.load_state_dict(state_dict, strict=False)
                elif args.init.lower() == "gmml":
                    model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    model = get_prepared_checkpoint(model, "models/gmml_1000e_nih.pth")
                    #model = LabelTokenViT(args.num_class, model, label_layers=args.label_layers)
                elif args.init.lower() =="deit":
                    model = timm.create_model('deit_small_patch16_224', num_classes=args.num_class, pretrained=True)           

            elif args.model_name.lower() == "swin_base": 
                if args.init.lower() =="random":
                    model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=False)
                elif args.init.lower() =="imagenet_21kto1k":
                    model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, pretrained=True)
                elif args.init.lower() =="imagenet_21k":
                    model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=True)
                
            elif args.model_name.lower() == "swin_tiny": 
                if args.init.lower() =="random":
                    model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=False)
                elif args.init.lower() =="imagenet_1k" or args.init.lower() =="imagenet":
                    model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=True)
            
        elif os.path.isfile(args.proxy_dir):
            print("Creating model from pretrained weights: "+ args.proxy_dir)
            if args.model_name.lower() == "vit_base":
                if args.init.lower() == "simmim":
                    model = simmim.create_model(args)
                else:
                    model = VisionTransformer(num_classes=args.num_class,
                            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    model.default_cfg = _cfg()
                    load_proxy_dir(model, args.init.lower(), args.proxy_dir)
                
            elif args.model_name.lower() == "vit_small":
                num_heads = 6
                if "moco" in args.init.lower():
                    num_heads = 6
                model = VisionTransformer(in_chans=args.nc, num_classes=args.num_class,
                        patch_size=16, embed_dim=384, depth=12, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                load_proxy_dir(model, args.init.lower(), args.proxy_dir) 
                
            elif args.model_name.lower() == "swin_base":
                if args.init.lower() == "simmim":
                    model = simmim.create_model(args)
                elif args.init.lower() =="imagenet_1k" or args.init.lower() =="imagenet":
                    model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, checkpoint_path=args.proxy_dir)
    
            elif args.model_name.lower() == "swin_tiny": 
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)
                load_proxy_dir(model, args.init.lower(), args.proxy_dir)
        else:
            raise FileNotFoundError(f"Proxy dir {args.proxy_dir} is not a file or directory!")
        if model is None:
            print("Not provide {} pretrained weights for {}.".format(args.init, args.model_name))
            raise Exception("Please provide correct parameters to load the model!")
    else:
        if args.init.lower() =="random" or args.init.lower() =="imagenet":
            model = ClassificationNet(args.model_name.lower(), args.num_class, args, weight=args.init,
                                activation=args.activate)

        else:
            model = ClassificationNet(args.model_name.lower(), args.num_class, args, weight=args.proxy_dir,
                                activation=args.activate)

    return model


def ClassificationNet(arch_name, num_class, args, conv=None, weight=None, activation=None):
    if "vit" not in arch_name.lower():
        if weight is None:
            weight = "none"

        if conv is None:
            try:
                model = resnet_wider.__dict__[arch_name](sobel=False)
            except:
                model = models.__dict__[arch_name](pretrained=False)
        else:
            if arch_name.lower().startswith("resnet"):
                model = resnet_wider.__dict__[arch_name + "_layerwise"](conv, sobel=False)
            elif arch_name.lower().startswith("densenet"):
                model = densenet.__dict__[arch_name + "_layerwise"](conv)

        if arch_name.lower().startswith("resnet"):
            kernelCount = model.fc.in_features
            if activation is None:
                model.fc = nn.Linear(kernelCount, num_class)
            elif activation == "Sigmoid":
                model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
            # init the fc layer
            if activation is None:
                model.fc.weight.data.normal_(mean=0.0, std=0.01)
                model.fc.bias.data.zero_()
            else:
                model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
                model.fc[0].bias.data.zero_()
        elif arch_name.lower().startswith("densenet"):
            kernelCount = model.classifier.in_features
            if activation is None:
                model.classifier = nn.Linear(kernelCount, num_class)
            elif activation == "Sigmoid":
                model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
            # init the classifier layer
            if activation is None:
                model.classifier.weight.data.normal_(mean=0.0, std=0.01)
                model.classifier.bias.data.zero_()
            else:
                model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
                model.classifier[0].bias.data.zero_()

        def _weight_loading_check(_arch_name, _activation, _msg):
            if len(_msg.missing_keys) != 0:
                if _arch_name.lower().startswith("resnet"):
                    if _activation is None:
                        assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
                    else:
                        assert set(_msg.missing_keys) == {"fc.0.weight", "fc.0.bias"}
                elif _arch_name.lower().startswith("densenet"):
                    if _activation is None:
                        assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
                    else:
                        assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}

        if weight.lower() == "imagenet":
            pretrained_model = models.__dict__[arch_name](pretrained=True)
            state_dict = pretrained_model.state_dict()

            # delete fc layer
            for k in list(state_dict.keys()):
                if k.startswith('fc') or k.startswith('classifier'):
                    del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded supervised ImageNet pre-trained model")
        elif os.path.isfile(weight):
            checkpoint = torch.load(weight, map_location="cpu")
            if "moco" in args.init.lower():
                state_dict = checkpoint
            else:
                state_dict = checkpoint["state_dict"]

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}

            for k in list(state_dict.keys()):
                if k.startswith('fc') or k.startswith('classifier') or k.startswith('projection_head') or k.startswith('prototypes'):
                    del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg.missing_keys)


        # reinitialize fc layer again
        if arch_name.lower().startswith("resnet"):
            if activation is None:
                model.fc.weight.data.normal_(mean=0.0, std=0.01)
                model.fc.bias.data.zero_()
            else:
                model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
                model.fc[0].bias.data.zero_()
        elif arch_name.lower().startswith("densenet"):
            if activation is None:
                model.classifier.weight.data.normal_(mean=0.0, std=0.01)
                model.classifier.bias.data.zero_()
            else:
                model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)

    else:
        if arch_name.lower() == "vit_base":
            model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class)
        elif arch_name.lower() == "vit_small":
            model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class)  
        elif arch_name.lower() == "swin_base": 
            model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class)
        elif arch_name.lower() == "swin_tiny": 
            model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)

    return model


def load_proxy_dir(model, init, proxy_dir):
    checkpoint = torch.load(proxy_dir, map_location="cpu")
    if init =="dino":
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in checkpoint:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = checkpoint[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    elif "moco" in init:
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            raise ValueError("No state_dict or model in checkpoint")
    elif init == "moby":
        state_dict = checkpoint['model']
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    elif init == "mae":
        state_dict = checkpoint['model']  
    elif ("gmml" in init) or ("sit" in init):
        state_dict = get_prepared_checkpoint(model, proxy_dir)
    else:
        print("Trying to load the checkpoint for {} at {}, but we cannot guarantee the success.".format(init, proxy_dir))
        state_dict = checkpoint["state_dict"]
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded with msg: {}'.format(msg))
    return model


def save_checkpoint(state,filename='model'):

    torch.save( state,filename + '.pth.tar')




