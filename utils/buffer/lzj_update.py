import torch
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.buffer_utils import ClassBalancedRandomSampling, random_retrieve
from utils.buffer.lzj_utils import compute_topsis
from utils.setup_elements import n_classes
from utils.utils import nonzero_indices, maybe_cuda


class LZJ_update(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = params.k
        self.mem_size = params.mem_size
        self.num_tasks = params.num_tasks
        self.out_dim = n_classes[params.data]
        self.n_smp_cls = int(params.n_smp_cls)
        self.n_total_smp = int(params.n_smp_cls * self.out_dim)
        self.reservoir_update = Reservoir_update(params)
        ClassBalancedRandomSampling.class_index_cache = None

    def update(self, buffer, x, y, **kwargs):
        model = buffer.model

        place_left = self.mem_size - buffer.current_index

        # If buffer is not filled, use available space to store whole or part of batch
        if place_left:
            x_fit = x[:place_left]
            y_fit = y[:place_left]

            ind = torch.arange(start=buffer.current_index, end=buffer.current_index + x_fit.size(0), device=self.device)
            ClassBalancedRandomSampling.update_cache(buffer.buffer_label, self.out_dim,
                                                     new_y=y_fit, ind=ind, device=self.device)
            self.reservoir_update.update(buffer, x_fit, y_fit)

        # If buffer is filled, update buffer by sv
        if buffer.current_index == self.mem_size:
            # remove what is already in the buffer
            cur_x, cur_y = x[place_left:], y[place_left:]
            self._update_by_topsis(model, buffer, cur_x, cur_y)

    def _update_by_topsis(self, model, buffer, cur_x, cur_y):
        """
            Returns indices for replacement.
            Buffered instances with smallest SV are replaced by current input with higher SV.
                Args:
                    model (object): neural network.
                    buffer (object): buffer object.
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                Returns
                    ind_buffer (tensor): indices of buffered instances to be replaced.
                    ind_cur (tensor): indices of current data to do replacement.
        """
        cur_x = maybe_cuda(cur_x)
        cur_y = maybe_cuda(cur_y)
        
        eval_x = buffer.buffer_img
        eval_y = buffer.buffer_label

        # Concatenate current input batch to candidate set
        cand_x = torch.cat((eval_x, cur_x))
        cand_y = torch.cat((eval_y, cur_y))
        # 计算eval_x的深度特征的中间，并根据距离给cand_x排序,返回排序后的cand_x 和 cand_y
        ret_x,ret_y = compute_topsis(model, cand_x, cand_y, cand_x, cand_y, device=self.device)


        buffer.n_seen_so_far += cur_x.size(0)

        # perform overwrite op
        y_upt = ret_y[:self.mem_size]
        x_upt = ret_x[:self.mem_size]
        #print("=====upd_size:",y_upt.size())
        buffer.buffer_img = x_upt
        buffer.buffer_label = y_upt
#----------- Avg_End_Acc (0.0324, nan) Avg_End_Fgt (0.215, nan) Avg_Acc (0.06917, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#decend False just update----------- Avg_End_Acc (0.0308, nan) Avg_End_Fgt (0.18810000000000002, nan) Avg_Acc (0.0626538888888889, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#decend False just retrieve----------- Avg_End_Acc (0.0228, nan) Avg_End_Fgt (0.1802, nan) Avg_Acc (0.06664063492063493, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#decend Ture----------- Avg_End_Acc (0.0160, nan) Avg_End_Fgt (0.103, nan) Avg_Acc (0.0566, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#decend False----------- Avg_End_Acc (0.0309, nan) Avg_End_Fgt (0.1419, nan) Avg_Acc (0.0593, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#e2 = random----------- Avg_End_Acc (0.0319, nan) Avg_End_Fgt (0.22409999999999997, nan) Avg_Acc (0.07884563492063493, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#e2 = our----------- Avg_End_Acc (0.0354, nan) Avg_End_Fgt (0.19249999999999998, nan) Avg_Acc (0.06257464285714286, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#e3r ----------- Avg_End_Acc (0.038900000000000004, nan) Avg_End_Fgt (0.26280000000000003, nan) Avg_Acc (0.08396162698412699, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#e3o----------- Avg_End_Acc (0.035199999999999995, nan) Avg_End_Fgt (0.1424, nan) Avg_Acc (0.05863170634920635, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
#down_sample5000----------- Avg_End_Acc (0.0317, nan) Avg_End_Fgt (0.20220000000000002, nan) Avg_Acc (0.07558043650793651, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------
# ----------- Avg_End_Acc (0.0303, nan) Avg_End_Fgt (0.189, nan) Avg_Acc (0.07141285714285714, nan) Avg_Bwtp (0.0, nan) Avg_Fwt (0.0, nan)-----------