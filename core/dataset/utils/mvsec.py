import os
import h5py
import numpy as np
import cv2


class MVSEC():
    def __init__(self, path, name):
        """
        :param path: HDF5文件夹根路径
        :param name: e.g. indoor_flying1
        """
        # Load path
        data_path = os.path.join(path, name + '_data.hdf5')
        gt_path = os.path.join(path, name + '_gt.hdf5')

        # Data
        data_raw = h5py.File(data_path, 'r')
        self.image_data = data_raw['davis']['left']['image_raw']
        self.image_ts = data_raw['davis']['left']['image_raw_ts']
        self.event_data = data_raw['davis']['left']['events']
        self.image_event_inds = data_raw['davis']['left']['image_raw_event_inds'] # index of event closet to image(i)

        # Ground Truth
        self.gt_raw = h5py.File(gt_path, 'r')
        self.gt_flow_data = self.gt_raw['davis']['left']['flow_dist']  # List:(N,2,H,W)
        self.gt_timestamps = self.gt_raw['davis']['left']['flow_dist_ts']

        # Length
        self.__len_i = len(self.image_data)
        self.__len_e = len(self.event_data)
        self.__len_gt = len(self.gt_flow_data)
        
    def get_image(self, index:int) -> np.ndarray:
        """
        :return greyimage: [260,346]
        """
        return self.image_data[index]

    def get_events(self, start:int, end:int) -> np.ndarray:
        """
        :return e: A [N,4] list of (x,y,t,p)  
          - (x,y) is in range of (260,346)
          - t is Unix timestampe
          - p = [1,-1]
        """
        return self.event_data[start:end]
        
    def get_flow(self, index:int) -> np.ndarray:
        """
        :returns flow: [2, 260, 346] = [(vx, vy), H, W]
        """
        return self.gt_flow_data[index]
    
    def get_time_ofimage(self, index:int) -> np.float64:
        return self.image_ts[index]
    
    def get_idx_imageToevent(self, index:int) -> np.int64:
        return self.image_event_inds[index]

    def estimate_flow(self, T_start:int, T_end:int) -> np.ndarray:
        """
        :returns flow: [260, 346, 2] = [H, W, (vx, vy)]
        """
        U_gt, V_gt = self.__estimate_corresponding_gt_flow(self.gt_flow_data, self.gt_timestamps, T_start, T_end)
        gt_flow = np.stack((U_gt, V_gt), axis=2)
        return gt_flow

    def len_image(self) -> int:
        return self.__len_i
    
    def len_event(self) -> int:
        return self.__len_e

    def len_flow(self) -> int:
        return self.__len_gt
    
    def __estimate_corresponding_gt_flow(self, flows, gt_timestamps, start_time, end_time):
        """
        :param flows: [N, 2, H, W]

        The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we
        need to propagate the ground truth flow over the time between two images.
        This function assumes that the ground truth flow is in terms of pixel displacement, not velocity.

        Pseudo code for this process is as follows:

        x_orig = range(cols)
        y_orig = range(rows)
        x_prop = x_orig
        y_prop = y_orig
        Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
        for all of these flows:
        x_prop = x_prop + gt_flow_x(x_prop, y_prop)
        y_prop = y_prop + gt_flow_y(x_prop, y_prop)

        The final flow, then, is x_prop - x-orig, y_prop - y_orig.
        Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.

        Inputs:
        x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at
            each timestamp.
        gt_timestamps - timestamp for each flow array.
        start_time, end_time - gt flow will be estimated between start_time and end time.
        """
        # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between gt_iter and gt_iter+1.
        gt_iter = np.searchsorted(gt_timestamps, start_time, side='right') - 1
        gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
        x_flow = np.squeeze(flows[gt_iter, 0, ...])
        y_flow = np.squeeze(flows[gt_iter, 1, ...])

        dt = end_time - start_time

        # No need to propagate if the desired dt is shorter than the time between gt timestamps.
        if gt_dt > dt:
            return x_flow * dt / gt_dt, y_flow * dt / gt_dt

        x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                        np.arange(x_flow.shape[0]))
        x_indices = x_indices.astype(np.float32)
        y_indices = y_indices.astype(np.float32)

        orig_x_indices = np.copy(x_indices)
        orig_y_indices = np.copy(y_indices)

        # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
        x_mask = np.ones(x_indices.shape, dtype=bool)
        y_mask = np.ones(y_indices.shape, dtype=bool)

        scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
        total_dt = gt_timestamps[gt_iter + 1] - start_time

        self.__prop_flow(x_flow, y_flow,
                x_indices, y_indices,
                x_mask, y_mask,
                scale_factor=scale_factor)

        gt_iter += 1

        while gt_timestamps[gt_iter + 1] < end_time:
            x_flow = np.squeeze(flows[gt_iter, 0, ...])
            y_flow = np.squeeze(flows[gt_iter, 1, ...])

            self.__prop_flow(x_flow, y_flow,
                             x_indices, y_indices,
                             x_mask, y_mask)
            total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

            gt_iter += 1

        final_dt = end_time - gt_timestamps[gt_iter]
        total_dt += final_dt

        final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        x_flow = np.squeeze(flows[gt_iter, 0, ...])
        y_flow = np.squeeze(flows[gt_iter, 1, ...])

        scale_factor = final_dt / final_gt_dt

        self.__prop_flow(x_flow, y_flow,
                x_indices, y_indices,
                x_mask, y_mask,
                scale_factor)

        x_shift = x_indices - orig_x_indices
        y_shift = y_indices - orig_y_indices
        x_shift[~x_mask] = 0
        y_shift[~y_mask] = 0

        return x_shift, y_shift
    
    
    def __prop_flow(self, x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
        """
        Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow.
        x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
        The optional scale_factor will scale the final displacement.
        """
        flow_x_interp = cv2.remap(x_flow,
                                x_indices,
                                y_indices,
                                cv2.INTER_NEAREST)

        flow_y_interp = cv2.remap(y_flow,
                                x_indices,
                                y_indices,
                                cv2.INTER_NEAREST)

        x_mask[flow_x_interp == 0] = False
        y_mask[flow_y_interp == 0] = False

        x_indices += flow_x_interp * scale_factor
        y_indices += flow_y_interp * scale_factor

        return
