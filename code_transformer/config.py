import torch
import yaml
from easydict import EasyDict as edict  # type: ignore

def create_config(args):
    cfg = edict()
    
    for k,v in args.items():
        cfg[k] = v

    with open(args['model_file'], 'r') as file:
        model_config = yaml.safe_load(file)
    
    for k,v in model_config.items():
        cfg[k] = v
    
    # Parse the task dictionary separately
    cfg.TASKS, extra_args = parse_task_dictionary(cfg['train_db_name'], cfg['task_dictionary'])

    for k, v in extra_args.items():
        cfg[k] = v
    
    # Other arguments   
    if cfg['train_db_name'] == 'PASCALContext':
        cfg.TRAIN = edict()
        cfg.TEST = edict()
        cfg.TRAIN.SCALE = (512, 512)
        cfg.TEST.SCALE = (512, 512)

    else:
        raise NotImplementedError
    
    from configs.mypath import db_paths, PROJECT_ROOT_DIR
    cfg['db_paths'] = db_paths
    cfg['PROJECT_ROOT_DIR'] = PROJECT_ROOT_DIR
    
    return cfg

def parse_task_dictionary(db_name, task_dictionary):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}

    if 'include_semseg' in task_dictionary.keys() and task_dictionary['include_semseg']:
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'PASCALContext':
            task_cfg.NUM_OUTPUT[tmp] = 21
        else:
            raise NotImplementedError

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        # Set effective depth evaluation range. Refer to:
        # https://github.com/sjsu-smart-lab/Self-supervised-Monocular-Trained-Depth-Estimation-using-Self-attention-and-Discrete-Disparity-Volum/blob/3c6f46ab03cfd424b677dfeb0c4a45d6269415a9/evaluate_city_depth.py#L55
        task_cfg.depth_max = 80.0
        task_cfg.depth_min = 0.

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'
        assert(db_name in ['PASCALContext', 'NYUD', 'nyuv2', 'cityscapes'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3

    if 'include_sal' in task_dictionary.keys() and task_dictionary['include_sal']:
        # Saliency Estimation
        assert(db_name == 'PASCALContext')
        tmp = 'sal'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 2

    if 'include_edge' in task_dictionary.keys() and task_dictionary['include_edge']:
        # Edge Detection
        assert(db_name == 'PASCALContext' or db_name == 'NYUD')
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        other_args['edge_w'] = task_dictionary['edge_w']

    if 'include_human_parts' in task_dictionary.keys() and task_dictionary['include_human_parts']:
        # Human Parts Segmentation
        assert(db_name == 'PASCALContext')
        tmp = 'human_parts'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 7

    return task_cfg, other_args


def get_backbone(opt):
    """ Return the backbone network """    
    if opt.backbone == 'swin_v2_t':
        from torchvision.models import swin_v2_t, Swin_V2_T_Weights
        swin_model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1, progress=True)
        backbone = torch.nn.Sequential(
            swin_model.features,
            swin_model.norm,
            swin_model.permute
        )
        backbone_channels = 768

    elif opt.backbone == 'swin_v2_s':
        from torchvision.models import swin_v2_s, Swin_V2_S_Weights
        swin_model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1, progress=True)
        backbone = torch.nn.Sequential(
            swin_model.features,
            swin_model.norm,
            swin_model.permute
        )
        backbone_channels = 768

    elif opt.backbone == 'swin_v2_b':
        from torchvision.models import swin_v2_b, Swin_V2_B_Weights
        swin_model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1, progress=True)
        backbone = torch.nn.Sequential(
            swin_model.features,
            swin_model.norm,
            swin_model.permute
        )
        backbone_channels = 1024

    else:
        raise NotImplementedError(f'{opt.backbone} not implimented')

    if 'fuse_hrnet' in opt['backbone_kwargs'] and opt['backbone_kwargs']['fuse_hrnet']: # Fuse the multi-scale HRNet features
        from models.hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels

def get_backbone_dims(opt):

    if opt.backbone == 'swin_v2_t' or opt.backbone == 'swin_v2_s' or opt.backbone == 'swin_v2_b':
        dims = [opt.TRAIN.SCALE[0]//32, opt.TRAIN.SCALE[1]//32]

    else:
        raise NotImplementedError(f'{opt.backbone} not implimented')

    return dims

def get_head(opt, backbone_channels, task):
    """ Return the decoder head """

    if opt['head'] == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, opt.TASKS.NUM_OUTPUT[task])

    elif opt['head'] == 'hrnet':
        from models.hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, opt.TASKS.NUM_OUTPUT[task])

    else:
        raise NotImplementedError

def get_model(opt):

    if opt.setup == 'single_task':
        
        backbones = {}
        for task in opt.TASKS.NAMES:
            backbone, backbone_channels = get_backbone(opt)
            backbones[task] = backbone
        backbones = torch.nn.ModuleDict(backbones)
        
        heads = torch.nn.ModuleDict({task: get_head(opt, backbone_channels, task) for task in opt.TASKS.NAMES})

        from models.baseline import STL
        model = STL(opt, backbones, backbone_channels, heads)

    
    elif opt.setup == 'multi_task':
        
        backbone, backbone_channels = get_backbone(opt)

        if opt.model == 'hps':
            from models.baseline import HPS
            heads = torch.nn.ModuleDict({task: get_head(opt, backbone_channels, task) for task in opt.TASKS.NAMES})
            model = HPS(opt, backbone, backbone_channels, heads)

        elif opt.model == 'padnet':
            from models.padnet import PADNet
            heads = torch.nn.ModuleDict({task: get_head(opt, 256, task) for task in opt.TASKS.NAMES})
            model = PADNet(opt, backbone, backbone_channels, heads)
        
        elif opt.model == 'papnet':
            from models.papnet import PAPNet
            heads = torch.nn.ModuleDict({task: get_head(opt, 128, task) for task in opt.TASKS.NAMES})
            model = PAPNet(opt, backbone, backbone_channels, heads)
        
        elif opt.model == 'ctal':
            from models.ctal import CTALNet
            backbone_dims = get_backbone_dims(opt)
            heads = torch.nn.ModuleDict({task: get_head(opt, 128, task) for task in opt.TASKS.NAMES})
            model = CTALNet(opt, backbone, backbone_channels, backbone_dims, heads)
        
        else:
            raise NotImplementedError(f'{opt.model} not implimeneted')
    
    else:
        raise NotImplementedError(f'Unknown setup: {opt.setup}')
    
    return model

""" 
    Loss functions 
"""

def get_loss(opt, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=opt['edge_w'], ignore_index=opt.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=opt.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=opt.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=opt.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(ignore_invalid_area=opt.ignore_invalid_area_depth, ignore_index=-1)

    else:
        criterion = None

    return criterion

def get_criterion(opt):
    if 'loss_weighting' in opt['loss_kwargs']:
        loss_weighting = get_loss_weighting(opt)

    if opt['loss_kwargs']['loss_scheme'] == 'stl':
        from losses.loss_schemes import SingleTaskLoss
        losses = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        return SingleTaskLoss(opt.TASKS.NAMES, losses)

    elif opt['loss_kwargs']['loss_scheme'] == 'hps':
        from losses.loss_schemes import MultiTaskLoss
        losses = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        return MultiTaskLoss(opt.TASKS.NAMES, losses, loss_weighting)

    elif opt['loss_kwargs']['loss_scheme'] == 'padnet':
        from losses.loss_schemes import PADNetLoss
        losses = {}
        losses['initial'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        losses['final'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        return PADNetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, losses, loss_weighting)
    
    elif opt['loss_kwargs']['loss_scheme'] == 'papnet':
        from losses.loss_schemes import PAPNetLoss
        losses = {}
        losses['initial'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        losses['final'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        return PAPNetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, losses, loss_weighting)

    elif opt['loss_kwargs']['loss_scheme'] == 'ctal':
        from losses.loss_schemes import CTALLoss
        losses = {}
        losses['initial'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        losses['final'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        return CTALLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, losses, loss_weighting)
    
    elif opt['loss_kwargs']['loss_scheme'] == 'mti_net':
        from losses.loss_schemes import MTINetLoss
        losses = {}
        losses['initial'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        losses['final'] = {t: get_loss(opt, t) for t in opt.TASKS.NAMES}
        return MTINetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, losses, loss_weighting)

    else:
        raise NotImplementedError(f'{opt.loss_scheme} not implimented')
    
def get_loss_weighting(opt):
    if opt['loss_kwargs']['loss_weighting'] == 'scalarization':
        from losses.loss_weights import Scalarization
        return Scalarization(opt)
    
    elif opt['loss_kwargs']['loss_weighting'] == 'uncertainty':
        from losses.loss_weights import Uncertainty
        return Uncertainty(opt)

    else:
        raise NotImplementedError

def get_transformations(opt):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if opt['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=opt.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=opt.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=opt.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None
    
def get_train_dataset(opt, transform=None):
    """ Return the train dataset """

    db_name = opt['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(opt.db_paths['PASCALContext'],
                         do_semseg='semseg' in opt.TASKS.NAMES,
                         do_normals='normals' in opt.TASKS.NAMES,
                         do_edge='edge' in opt.TASKS.NAMES,
                         do_human_parts='human_parts' in opt.TASKS.NAMES,
                         do_sal='sal' in opt.TASKS.NAMES,
                         split='train',
                         transform=transform)
        
    else:
        raise NotImplemented("train_db_name: Choose must be PASCALContext")
    
    return database

def get_test_dataset(opt, transform=None):
    """ Return the test dataset """

    db_name = opt['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(opt.db_paths['PASCALContext'],
                         do_semseg='semseg' in opt.TASKS.NAMES,
                         do_normals='normals' in opt.TASKS.NAMES,
                         do_edge='edge' in opt.TASKS.NAMES,
                         do_human_parts='human_parts' in opt.TASKS.NAMES,
                         do_sal='sal' in opt.TASKS.NAMES,
                         split='val',
                         transform=transform)
        
    else:
        raise NotImplemented("val_db_name: Choose must be PASCALContext")

    return database