
import torch
from torch import nn

from .modules import BasicEncoder, SmallEncoder, EVEN, BasicMotionEncoder, CorrBlock, BasicUpdateBlock
from .utils import initialize_flow, upflow8, upsample_flow


class E2Flow(nn.Module):
    def __init__(self, cfg):
        super(E2Flow, self).__init__()
        self.cfg = cfg
        self.hidden_dim = hdim = self.cfg.hidden_dim
        self.context_dim = cdim = self.cfg.context_dim

        # 1. Low Feature
        self.img_encoder = BasicEncoder(input_dim=3, output_dim=256, norm_fn='instance', dropout=self.cfg.dropout)
        self.event_encoder_f = SmallEncoder(input_dim=self.cfg.events_dim, output_dim=128, norm_fn='instance', dropout=self.cfg.dropout)
        self.event_enhance_feature = EVEN(event_dim=128, image_dim=256)

        # 2. Motion Feature
        self.event_encoder_m = SmallEncoder(input_dim=self.cfg.events_dim , output_dim=128, norm_fn='instance', dropout=self.cfg.dropout)
        self.event_enhance_motion = EVEN(event_dim=self.cfg.corr_levels * (2 * self.cfg.corr_radius + 1) ** 2, 
                                         image_dim=self.cfg.corr_levels * (2 * self.cfg.corr_radius + 1) ** 2)

        # 3. Context Feature
        self.context_encoder = BasicEncoder(input_dim=3, output_dim=hdim + cdim, norm_fn='batch', dropout=self.cfg.dropout)
        self.event_encoder_c = SmallEncoder(input_dim=self.cfg.events_dim, output_dim=128, norm_fn='batch', dropout=self.cfg.dropout)
        self.event_enhance_context = EVEN(event_dim=128, image_dim=hdim + cdim)

        # 4. Updator
        self.update_block = BasicUpdateBlock(cfg=cfg, context_dim=cdim, hidden_dim=hdim)

    def forward(self, image_0, image_1, events):
        """
        @param image: (Batchs, Channels, H, W)
        @param events: (Batchs, Bins, H, W)
        """
        # ----------------------------------------- 0. Preprocess --------------------------------------------------
        image_0 = 2 * (image_0 / 255.0) - 1.0
        image_1 = 2 * (image_1 / 255.0) - 1.0

        # ----------------------------------------- 1. Cal Features -----------------------------------------------
        # Cal feature of image
        fmap1, fmap2 = self.img_encoder([image_0, image_1])

        # Cal feature of event
        f_event_1 = self.event_encoder_f(events)
        f_event_2 = self.event_encoder_m(events)
        f_event_3 = self.event_encoder_c(events)

        # Feature Fusion
        fmap1 = self.event_enhance_feature(fmap1, f_event_1)
        fmap2 = self.event_enhance_feature(fmap2, f_event_2)
        
         # Cal feature of context
        context_init = self.context_encoder(image_0)
        context_init = self.event_enhance_context(context_init, f_event_3)

        # Cal feature for GRU
        net, ctx = torch.split(context_init, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        ctx = torch.relu(ctx)

        # Init Correlation
        img_corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        evt_corr_fn = CorrBlock(f_event_1, f_event_2, num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)

        # ----------------------------------------- 2. Init Flow --------------------------------------------------
        coords0, coords1 = initialize_flow(image_0)
        # ----------------------------------------- 3. GRU iteration ---------------------------------------------
        flow_predictions = []
        for itr in range(self.cfg.iters):
            # Lookup of image feature
            coords1 = coords1.detach()
            flow = coords1 - coords0

            # Motion feature
            f_corr = img_corr_fn(coords1)
            evt_lookup = evt_corr_fn(coords1)
            f_corr = self.event_enhance_motion(f_corr, evt_lookup)

            # GRU
            net, up_mask, delta_flow = self.update_block(net, ctx, f_corr, flow)

            # Finetune flow
            coords1 = coords1 + delta_flow

            # upsample predictions
            flow_up = upflow8(coords1 - coords0) if up_mask is None else upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
        
        # ----------------------------------------------- 4. End ---------------------------------------------------
        return flow_predictions
        # return flow_predictions[-1]
