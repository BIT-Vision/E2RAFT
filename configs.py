from yacs.config import CfgNode as CN

_CN = CN()
# Base setting
_CN.seed = 0
_CN.voxel_bins = 10

# For Training
_CN.training = CN()
_CN.training.checkpoint = None
_CN.training.save_path = './save'
_CN.training.prefix = 'e2flow'
_CN.training.epochs = 120
_CN.training.batch_size = 10
_CN.training.learning_rate = 4e-4
_CN.training.pct_start = 0.1

_CN.training.adamw_decay = 1e-4
_CN.training.epsilon = 1e-8
_CN.training.mixed_precision = True
_CN.training.grad_clip = 1.0

# For testing
_CN.testing = CN()
_CN.testing.checkpoint = './checkpoints/e2flow/e2flow-chairsDark.pth'
_CN.testing.save_path = "./outcomes"

# For Loss
_CN.loss = CN()
_CN.loss.gamma = 0.8

# For Dataset
_CN.dataset = CN()
_CN.dataset.FlyingChairs = CN()
_CN.dataset.FlyingChairs.data_path = "/ssd/zhangpengjie/FlyingChairs2" # "./datasets/FlyingChairs2"
_CN.dataset.FlyingChairs.IF_dark = True
_CN.dataset.FlyingChairs.crop_size = [368, 496] # Inital=(384,512)
_CN.dataset.FlyingChairs.do_flip = True
_CN.dataset.FlyingChairs.voxel_bins = _CN.voxel_bins

_CN.dataset.MVSEC = CN()
_CN.dataset.MVSEC.data_path = "/ssd/zhangpengjie/MVSEC" # "./datasets/MVSEC"
_CN.dataset.MVSEC.voxel_bins = _CN.voxel_bins
_CN.dataset.MVSEC.crop_size = [256, 256] # Inital=(260,346)
_CN.dataset.MVSEC.do_flip = True
_CN.dataset.MVSEC.dt = 1

_CN.dataset.RealData = CN()
_CN.dataset.RealData.data_path = '/data/zhangpengjie/zhangpengjie/Workspace/Datasets/RealDataset' # "./datasets/RealData"
_CN.dataset.RealData.voxel_bins = _CN.voxel_bins
_CN.dataset.RealData.crop_size_indoor = [256, 320]   # Inital = (260,346,3)
_CN.dataset.RealData.crop_size_outdoor = [576, 1024] # Inital = (576, 1024, 3)

# For Model
_CN.model = CN()
_CN.model.events_dim = _CN.voxel_bins
_CN.model.iters = 12
_CN.model.dropout = 0
_CN.model.hidden_dim = 128
_CN.model.context_dim = 128
_CN.model.corr_radius = 4
_CN.model.corr_levels = 4

_CN.model.voxel_bins= _CN.voxel_bins    # For DCEI
_CN.model.mixed_precision = True # For DCEI

def get_cfg():
    return _CN.clone()
