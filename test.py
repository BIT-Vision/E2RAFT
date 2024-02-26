import argparse
import imageio
import numpy as np
import torch
import os
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_cfg
from core.model.e2flow.e2flow import E2Flow
from core.model.raft.raft import RAFT
from core.model.flowformer import default
from core.model.flowformer.LatentCostFormer.transformer import FlowFormer
from core.model.dcei.DCEIFlow import DCEIFlow

from core.dataset.FlyingChairsDark import FlyingChairsData
from core.dataset.MVSEC import MVSECData
from core.dataset.RealDarkData import RealDataset

from core.metric import Metric
from core.utils.EventProcess.EventToImage import voxel_to_rgb
from core.utils.FlowVisualization.FlowToImage import flow_to_image

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester():
    def __init__(self, cfgs, model) -> None:
        self.cfgs = cfgs

        self.model = model.eval()

        if not os.path.exists(self.cfgs.testing.save_path):
            os.makedirs(self.cfgs.testing.save_path)

        # Load Metric
        self.metric = Metric('numpy')
        self.sparse = True

        self.epe_list = []
        self._1pe_list = []
        self._2pe_list = []
        self._3pe_list = []
        self.outlier_list = []
        if self.sparse:
            self.epe_list_sparse = []
            self._1pe_list_sparse = []
            self._2pe_list_sparse = []
            self._3pe_list_sparse = []
            self.outlier_list_sparse = []
    
    
    def metric_push(self, flow_real, flow_predicted, valid, voxel):
        metric = self.metric
        # EPE
        # Get Flow Mask
        mag = metric.Magnitude(flow_real - flow_predicted)
        mask = (valid >= 0.5) & (mag < 400)
        # Get Event Mask
        event_mask = np.linalg.norm(voxel.cpu().detach().numpy(), ord=2, axis=1, keepdims=False) > 0
        sparse_mask = np.logical_and(mask, event_mask)

        # Dense
        epe = metric.EPE(flow_real, flow_predicted, mask)
        aee = metric.AEE(epe)
        self.epe_list.append(aee)

        _1pe = metric.NPE(epe, 1)
        self._1pe_list.append(_1pe)

        _2pe = metric.NPE(epe, 2)
        self._2pe_list.append(_2pe)

        _3pe = metric.NPE(epe, 3)
        self._3pe_list.append(_3pe)

        outlier = metric.Outlier(epe, metric.Magnitude(flow_real, mask))
        self.outlier_list.append(outlier)

        if self.sparse:
            # Sparse
            epe = metric.EPE(flow_real, flow_predicted, sparse_mask)
            aee = metric.AEE(epe)
            self.epe_list_sparse.append(aee)

            _1pe = metric.NPE(epe, 1)
            self._1pe_list_sparse.append(_1pe)

            _2pe = metric.NPE(epe, 2)
            self._2pe_list_sparse.append(_2pe)

            _3pe = metric.NPE(epe, 3)
            self._3pe_list_sparse.append(_3pe)

            outlier = metric.Outlier(epe, metric.Magnitude(flow_real, sparse_mask))
            self.outlier_list_sparse.append(outlier)
    
    def metric_summary(self):
        with open(os.path.join(self.cfgs.testing.save_path, "metric.txt"), 'w') as f:
            EPE = np.mean(self.epe_list)
            _1PE = np.mean(self._1pe_list)
            _2PE = np.mean(self._2pe_list)
            _3PE = np.mean(self._3pe_list)
            Outlier = np.mean(self.outlier_list)
            f.write('EPE=' + str(EPE) + '\n')
            f.write('1pe=' + str(_1PE) + '\n')
            f.write('2pe=' + str(_2PE) + '\n')
            f.write('3pe=' + str(_3PE) + '\n')
            f.write('Outlier=' + str(Outlier) + '\n')
            print('EPE=' + str(EPE))
            print('1pe=' + str(_1PE))
            print('2pe=' + str(_2PE))
            print('3pe=' + str(_3PE))
            print('Outlier=' + str(Outlier))              

            if self.sparse:
                EPE = np.mean(self.epe_list_sparse)
                _1PE = np.mean(self._1pe_list_sparse)
                _2PE = np.mean(self._2pe_list_sparse)
                _3PE = np.mean(self._3pe_list_sparse)
                Outlier = np.mean(self.outlier_list_sparse)
                f.write('EPE_sparse=' + str(EPE) + '\n')
                f.write('1pe_sparse=' + str(_1PE) + '\n')
                f.write('2pe_sparse=' + str(_2PE) + '\n')
                f.write('3pe_sparse=' + str(_3PE) + '\n')
                f.write('Outlier_sparse=' + str(Outlier) + '\n')
                print('EPE_sparse=' + str(EPE))
                print('1pe_sparse=' + str(_1PE))
                print('2pe_sparse=' + str(_2PE))
                print('3pe_sparse=' + str(_3PE))
                print('Outlier_sparse=' + str(Outlier))
    
    def visualization_save12(self, step, image_0, image_1, voxel):
        # 1. Images
        image_pre = image_0[0].cpu().detach().numpy().transpose(1, 2, 0)
        image_next = image_1[0].cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join(self.cfgs.testing.save_path, str(step)+"_preImage.png"), cv2.cvtColor(image_pre, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.cfgs.testing.save_path, str(step)+"_nextImage.png"), cv2.cvtColor(image_next, cv2.COLOR_RGB2BGR))

        # 2. Voxel
        voxel_rgb = voxel[0].cpu().detach().numpy()
        voxel_add = voxel_rgb[0]
        for j in range(1, self.cfgs.voxel_bins):
            if j < self.cfgs.voxel_bins / 2:
                voxel_add = voxel_add + voxel_rgb[j]
            else:
                voxel_add = voxel_add - voxel_rgb[j]
        voxel_rgb = voxel_to_rgb(voxel_add).transpose(1, 2, 0)
        cv2.imwrite(os.path.join(self.cfgs.testing.save_path, str(step)+"_Voxel.png"), cv2.cvtColor(voxel_rgb, cv2.COLOR_RGB2BGR))
    
    def visualization_save3(self, step, flow_real, flow_predicted):
        # 3. Flow
        predicted_rgb = flow_to_image(flow_predicted[0].transpose(1,2,0))
        real_rgb = flow_to_image(flow_real[0].transpose(1,2,0))
        cv2.imwrite(os.path.join(self.cfgs.testing.save_path, str(step)+"_PredFlow.png"), cv2.cvtColor(predicted_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.cfgs.testing.save_path, str(step)+"_RealFlow.png"), cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR))

    def test_f2(self, show=True):
        dataset_test = FlyingChairsData(self.cfgs.dataset.FlyingChairs, split='test', augmentor=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, 
                                     num_workers=1, worker_init_fn=np.random.seed(0), prefetch_factor=2)
        # Test loop
        step = 0
        for data_blob in tqdm(dataloader_test):
            # Input
            image_0, image_1, voxel, flow, valid = [x.to(device, non_blocking=True) for x in data_blob]

            if show:
                self.visualization_save12(image_0, image_1, voxel)

            # Predict
            flow_predictions = model(image_0, image_1, voxel)
            flow_predicted = flow_predictions[-1].cpu().detach().numpy()
            flow_real = flow.cpu().detach().numpy()
            valid = valid.cpu().detach().numpy()

            if show:
                self.visualization_save3(flow_real, flow_predicted)
            
            # Metric
            self.metric_push(flow_real, flow_predicted, valid, voxel)

            step = step + 1

        self.metric_summary()
    
    def test_mvsec(self, name='indoor_flying1', show=True):
        dataset_test = MVSECData(self.cfgs.dataset.MVSEC, name, augmentor=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, 
                                     num_workers=1, worker_init_fn=np.random.seed(0), prefetch_factor=2)
        Outdoor = True if name == 'outdoor_day1' or name == 'outdoor_day2' else False

        # Test Loop
        step = 0
        for data_blob in tqdm(dataloader_test):
            # Input
            image_0, image_1, voxel, flow, valid = [x.to(device, non_blocking=True) for x in data_blob]

            if show:
               self.visualization_save12(image_0, image_1, voxel)

            # Predict
            flow_predictions = model(image_0, image_1, voxel)
            flow_predicted = flow_predictions[-1].cpu().detach().numpy()
            flow_real = flow.cpu().detach().numpy()
            valid = valid.cpu().detach().numpy()

            if show:
               self.visualization_save3(flow_real, flow_predicted)
            
            if Outdoor:
                flow_real = flow_real[:, :, 0:190, :]
                flow_predicted = flow_predicted[:, :, 0:190, :]
                valid = valid[:, 0:190, :]
                voxel = voxel[:, :, 0:190, :]

            # Metric
            self.metric_push(flow_real, flow_predicted, valid, voxel)

            step = step + 1

        self.metric_summary()

    def test_real(self, name, show=True):
        dataset_test = RealDataset(self.cfgs.dataset.RealData, name)
        dataloader_test = DataLoader(dataset_test, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, 
                                     num_workers=1, worker_init_fn=np.random.seed(0), prefetch_factor=2)

        # Test Loop
        step = 0
        for data_blob in tqdm(dataloader_test):
            # Input
            image_0, image_1, voxel = [x.to(device, non_blocking=True) for x in data_blob]

            if show:
                self.visualization_save12(image_0, image_1, voxel)

            # Predict
            flow_predictions = model(image_0, image_1, voxel)
            flow_predicted = flow_predictions[-1].cpu().detach().numpy()

            if show:
                # 3. Flow
                predicted_rgb = flow_to_image(flow_predicted[0].transpose(1,2,0))
                cv2.imwrite(os.path.join(self.cfgs.testing.save_path, str(step)+"_PredFlow.png"), cv2.cvtColor(predicted_rgb, cv2.COLOR_RGB2BGR))
            
            step = step + 1
            

def setSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', help="model name", default='e2flow')
    parser.add_argument('--checkpoint', help="restore checkpoint")
    parser.add_argument('--datasetName', help="dataset for evaluation")
    parser.add_argument('--sequenceName', help="sequence for evaluation")
    args = parser.parse_args()

    # 0. Get configs
    cfgs = get_cfg()
    cfgs.update(vars(args))
    setSeed(cfgs.seed)

    # 1. Load model
    if cfgs.modelName == 'e2flow':
        model = E2Flow(cfgs.model).to(device)
    elif cfgs.modelName == 'raft':
        model = RAFT(cfgs.model).to(device)
    elif cfgs.modelName == 'flowformer':
        model = FlowFormer(default.get_cfg().latentcostformer)
    elif cfgs.modelName == 'dcei':
        model = DCEIFlow(cfgs.model, split='test').to(device)  # iter=6/12

    # 2. Load Checkpoint
    if cfgs.modelName == 'e2flow':
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(cfgs.testing.checkpoint)['model_state_dict'])
    elif cfgs.modelName == 'raft':
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(cfgs.testing.checkpoint))
    elif cfgs.modelName == 'flowformer':
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(cfgs.testing.checkpoint))
    elif cfgs.modelName == 'dcei':
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(cfgs.testing.checkpoint)['model_state_dict'])
    
    # 3. Run
    tester = Tester(cfgs, model)
    with torch.no_grad():
        if cfgs.datasetName == 'f2':
            tester.test_f2(show=True)
        elif cfgs.datasetName == 'mvsec':
            tester.test_mvsec(cfgs.sequenceName, show=True)
        elif cfgs.datasetName == 'real':
            tester.test_real(cfgs.sequenceName, show=True)
