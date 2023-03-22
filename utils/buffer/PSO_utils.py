import torch
from utils.utils import maybe_cuda, mini_batch_deep_features, euclidean_distance, nonzero_indices, ohe_label
from utils.setup_elements import n_classes
from utils.buffer.buffer_utils import ClassBalancedRandomSampling

def compute_mean_feature(buffer,x,y):
    model = buffer.model
    deep_features_ = mini_batch_deep_features(model, x, x.size(0))
    if buffer.category_center == None:
        buffer.category_center = torch.tensor([[0.0]*deep_features_.size(1)]*100)
    if buffer.all_center == None:
        buffer.all_center = torch.tensor([0.0]*deep_features_.size(1))

    buffer.all_center = (buffer.all_center * buffer.n_seen_so_far
                          + deep_features_.mean(0).to(buffer.all_center.device) * x.size(0)
                          )/(buffer.n_seen_so_far+x.size(0))
    for i in set(y):
        f_category = deep_features_[y==i]    
        _category_center = (f_category.sum(0).to(buffer.category_center.device) + 
                                       buffer.category_center[i-1].to(buffer.category_center.device)
                                        *buffer.n_category_seen_so_far[i-1])/(f_category.size(0)+buffer.n_category_seen_so_far[i-1])

        # print("_category_center",_category_center)
        buffer.category_center[i-1] = _category_center
        # print("buffer.category_center[i-1]",buffer.category_center[i-1])
        
    # print("buffer.all_center",buffer.all_center)
    # print("buffer.category_center",buffer.category_center)

def compute_PSO(buffer,model, cand_x, cand_y, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_cand = cand_x.size(0)
    # print("========n_eval:",n_eval)
    # print("========n_cand:",n_cand)
    # Initialize SV matrix to matrix of -1
    # Get deep features
    cand_df = mini_batch_deep_features(model, cand_x, n_cand)
    # print("========eval_df:",eval_df.size())
    # print("========cand_df:",cand_df.size())
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(buffer,cand_df,cand_y, n_cand)
    
    # print("sorted_ind_mat ",sorted_ind_mat)

    return cand_x[sorted_ind_mat],cand_y[sorted_ind_mat]



def sorted_cand_ind(buffer,cand_df, cand_y,n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    # print("========eval_df_mean:",eval_df_mean.size())
    # Compute distance between evaluation and candidate feature vectors
    category_center = torch.tensor([[0.0]*buffer.category_center.size(1)]*cand_df.size(0))
    for i in range(cand_df.size(0)):
        category_center[i] = buffer.category_center[cand_y[i]-1]
    distance_vector_category = euclidean_distance(category_center.to(cand_df.device), cand_df)
    distance_vector_center = euclidean_distance(buffer.all_center.to(cand_df.device), cand_df)
    # print("========distance_vector:",distance_vector)
    # Turn distance vector into distance matrix
    # Sort candidate set indices based on distance
    # True为选远的
    # 这里我们希望距离越小的排名越高，所以是升序,0是排序后的原列，1是序号
    d_ca=distance_vector_category.sort(0,False)[1]
    # print("d_ca",d_ca)
    d_ce=distance_vector_center.sort(0,False)[1]
    # print("d_ce",d_ce)
    sorted_cand_ind_ = (0.5*d_ca+0.5*d_ce).argsort(descending=False)
    # print("========sorted_cand_ind_:",sorted_cand_ind_)
    return sorted_cand_ind_


def add_minority_class_input(cur_x, cur_y, mem_size, num_class):
    """
    Find input instances from minority classes, and concatenate them to evaluation data/label tensors later.
    This facilitates the inclusion of minority class samples into memory when ASER's update method is used under online-class incremental setting.

    More details:

    Evaluation set may not contain any samples from minority classes (i.e., those classes with very few number of corresponding samples stored in the memory).
    This happens after task changes in online-class incremental setting.
    Minority class samples can then get very low or negative KNN-SV, making it difficult to store any of them in the memory.

    By identifying minority class samples in the current input batch, and concatenating them to the evaluation set,
        KNN-SV of the minority class samples can be artificially boosted (i.e., positive value with larger magnitude).
    This allows to quickly accomodate new class samples in the memory right after task changes.

    Threshold for being a minority class is a hyper-parameter related to the class proportion.
    In this implementation, it is randomly selected between 0 and 1 / number of all classes for each current input batch.


        Args:
            cur_x (tensor): current input data tensor.
            cur_y (tensor): current input label tensor.
            mem_size (int): memory size.
            num_class (int): number of classes in dataset.
        Returns
            minority_batch_x (tensor): subset of current input data from minority class.
            minority_batch_y (tensor): subset of current input label from minority class.
"""
    # Select input instances from minority classes that will be concatenated to pre-selected data
    threshold = torch.tensor(1).float().uniform_(0, 1 / num_class).item()

    # If number of buffered samples from certain class is lower than random threshold,
    #   that class is minority class
    cls_proportion = ClassBalancedRandomSampling.class_num_cache.float() / mem_size
    minority_ind = nonzero_indices(cls_proportion[cur_y] < threshold)

    minority_batch_x = cur_x[minority_ind]
    minority_batch_y = cur_y[minority_ind]
    return minority_batch_x, minority_batch_y
