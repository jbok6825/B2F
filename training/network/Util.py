import numpy as np
import torch
import torch.nn.functional as F
from process_dataset.Constant import *
import math

### Util Function for Preprocessing

# Temporal Bias

def compute_kl_weight(current_epoch, total_epochs, max_weight=0.3, 
                      warmup_ratio=0.25, hold_ratio=0.25):
    """
    Args:
        current_epoch (int): 현재 epoch
        total_epochs (int): 전체 epoch 수
        max_weight (float): KL 최대 가중치
        warmup_ratio (float): KL weight를 선형 증가시키는 비율 (예: 0.25 → 25% 동안 증가)
        hold_ratio (float): max_weight 상태를 유지하는 비율 (예: 0.25 → 그다음 25% 유지)

    Returns:
        float: KL weight at current epoch
    """
    warmup_epochs = int(total_epochs * warmup_ratio)
    hold_epochs = int(total_epochs * hold_ratio)
    decay_start = warmup_epochs + hold_epochs

    if current_epoch < warmup_epochs:
        return (current_epoch / warmup_epochs) * max_weight
    elif current_epoch < decay_start:
        return max_weight
    else:
        decay_epochs = total_epochs - decay_start
        decay_progress = (current_epoch - decay_start) / max(decay_epochs, 1)
        return max_weight * (1 - decay_progress)

def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S).to(device)
    for i in range(T):
        mask[i, i] = 0
    return (mask == 1).to(device=device)




def rotation_vector_to_matrix(rotation_vectors):
    """ 
    Convert rotation vectors to rotation matrices using Rodrigues' formula.
    This is implemented purely using PyTorch for GPU acceleration.
    
    Parameters:
        rotation_vectors (torch.Tensor): Rotation vectors with shape (batch_size, num_frames, 3).
    
    Returns:
        torch.Tensor: Rotation matrices with shape (batch_size, num_frames, 3, 3).
    """
    batch_size, num_frames, _ = rotation_vectors.shape
    theta = torch.norm(rotation_vectors, dim=-1, keepdim=True)  # Shape: (batch_size, num_frames, 1)

    # When theta is close to zero, the rotation matrix should be close to identity
    eps = 1e-5
    k = torch.zeros_like(rotation_vectors)  # Placeholder for normalized rotation vector

    # Normalize rotation vector when theta is greater than eps
    k = torch.where(theta > eps, rotation_vectors / theta, k)
    k = k.unsqueeze(-1)  # Shape: (batch_size, num_frames, 3, 1)
    
    # Rodrigues' rotation formula components
    I = torch.eye(3, device=rotation_vectors.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, 3, 3)  # Identity matrix expanded to match batch size and num_frames
    K = torch.zeros((batch_size, num_frames, 3, 3), device=rotation_vectors.device)  # Initialize K with correct shape
    
    K[..., 0, 1] = -k[..., 2, 0]
    K[..., 0, 2] = k[..., 1, 0]
    K[..., 1, 0] = k[..., 2, 0]
    K[..., 1, 2] = -k[..., 0, 0]
    K[..., 2, 0] = -k[..., 1, 0]
    K[..., 2, 1] = k[..., 0, 0]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    rotation_matrices = I + sin_theta[..., None] * K + (1 - cos_theta[..., None]) * torch.matmul(K, K)

    # For very small theta, approximate the rotation matrix as identity to avoid numerical instability
    theta_expanded = theta.squeeze(-1).expand_as(rotation_matrices[..., 0, 0])  # Ensure same shape by squeezing and expanding
    rotation_matrices = torch.where(theta_expanded[..., None, None] > eps, rotation_matrices, I)

    return rotation_matrices

def matrix_to_rotation_vector(rotation_matrices):
    """
    Convert rotation matrices to rotation vectors using the axis-angle representation.
    
    Parameters:
        rotation_matrices (torch.Tensor): Rotation matrices with shape (batch_size, num_frames, 3, 3).
    
    Returns:
        torch.Tensor: Rotation vectors with shape (batch_size, num_frames, 3).
    """
    R = rotation_matrices
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to avoid NaNs due to numerical errors
    
    theta = torch.acos(cos_theta)
    sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # sin(theta) = sqrt(1 - cos(theta)^2)
    
    # Compute the rotation vector
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    
    r = torch.stack([rx, ry, rz], dim=-1)  # Shape: (batch_size, num_frames, 3)
    
    # Avoid division by zero by using where with sin(theta)
    eps = 1e-6
    rotation_vectors = torch.where(sin_theta[..., None] > eps, r / (2 * sin_theta[..., None]) * theta[..., None], torch.zeros_like(r))
    
    return rotation_vectors

def compute_rotational_velocity_from_vectors(rotation_vectors, frame_duration=1/30):
    """
    Compute rotational velocity from rotation vectors using GPU-accelerated PyTorch operations.
    
    Parameters:
        rotation_vectors (torch.Tensor): Rotation vectors with shape (batch_size, num_frames, 3).
        frame_duration (float): Duration of each frame in seconds (default is 1/30 for 30 FPS).
    
    Returns:
        torch.Tensor: Rotational velocities with shape (batch_size, num_frames-1, 3).
    """
    # Convert rotation vectors to rotation matrices
    rotation_matrices = rotation_vector_to_matrix(rotation_vectors)
    
    # Calculate relative rotation matrices between consecutive frames
    relative_rotations = torch.matmul(rotation_matrices[:, 1:], rotation_matrices[:, :-1].transpose(-1, -2))
    
    # Convert relative rotations back to rotation vectors
    relative_rotation_vectors = matrix_to_rotation_vector(relative_rotations)
    
    # Compute rotational velocities
    rotational_velocities = relative_rotation_vectors / frame_duration
    
    return rotational_velocities


def rotate_matrix(theta, axis):
    if axis == 'X':
        mat = np.array([[1., 0., 0.],
                        [0., np.cos(theta), -np.sin(theta)],
                        [0., np.sin(theta), np.cos(theta)]])
    elif axis == 'Y':
        mat = np.array([[np.cos(theta), 0., np.sin(theta)],
                        [0., 1., 0.],
                        [-np.sin(theta), 0., np.cos(theta)]])
    elif axis == 'Z':
        mat = np.array([[np.cos(theta), -np.sin(theta), 0.],
                        [np.sin(theta), np.cos(theta), 0.],
                        [0., 0., 1.]])
    else:
        mat = np.identity(3)

    return mat

def triplet_loss(anchor, positive, negative, margin_triplet = None):

    if margin_triplet == None:
        margin_triplet = MARGIN_TRIPLET

    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = torch.mean(F.relu(pos_dist - neg_dist + margin_triplet))
    return loss


def l2distance_loss(vec1, vec2, epsilon=1e-8):
    # 정규화: feature_dim 차원에서
    vec1 = F.normalize(vec1, p=2, dim=-1)
    vec2 = F.normalize(vec2, p=2, dim=-1)
    
    # 두 벡터 간의 차이를 계산
    difference = vec1 - vec2
    
    # L2 norm 계산 (feature_dim 차원에서 제곱합의 제곱근)
    # 숫자 불안정성 방지를 위해 작은 값 epsilon 추가
    l2_norm = torch.sqrt(torch.sum(difference ** 2, dim=-1) + epsilon)  # (batch_size, frame_length) 또는 (batch_size,)
    
    # 손실 값을 평균
    if l2_norm.dim() == 2:  # (batch_size, frame_length)
        # frame_length 방향으로 평균 후, 배치 차원에서 평균
        loss = torch.mean(torch.mean(l2_norm, dim=1))
    elif l2_norm.dim() == 1:  # (batch_size,)
        # 배치 차원에서 평균
        loss = torch.mean(l2_norm)
    else:
        raise ValueError("Unsupported input dimensions.")
    
    return loss

# def entropy_loss(encoding_indices, num_embeddings):
#     # 1차원으로 변환 (flatten)
#     encoding_indices = encoding_indices.flatten()
    
#     # 정수형 변환 (long 타입)
#     encoding_indices = encoding_indices.long()
    
#     # 각 코드가 사용된 횟수를 계산
#     code_usage = torch.bincount(encoding_indices, minlength=num_embeddings).float()
    
#     # 사용된 코드들의 확률을 구하기 위해 normalize
#     code_usage = code_usage / code_usage.sum()
    
#     # 엔트로피 계산 (log2를 사용하는 정보 엔트로피)
#     entropy = -torch.sum(code_usage * torch.log2(code_usage + 1e-10))
    
#     return torch.exp(-entropy)  # 엔트로피를 최대화해야 하므로 -를 붙여서 minimize 하는 방향으로



def index_loss(vec1, vec2):
    loss = (vec1 != vec2).float()

    return torch.mean(loss)
    



def supervisedContrastive_loss(embeddings, labels, temperature = 0.1):
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    # Compute the similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T)

    # Create the positive mask
    labels = labels.unsqueeze(1)
    positive_mask = torch.eq(labels, labels.T).float()

    # Create the negative mask
    negative_mask = torch.ne(labels, labels.T).float()

    # Compute the positive and negative similarity
    sim_ij = torch.exp(sim_matrix / temperature)
    exp_sim_matrix = torch.exp(sim_matrix / temperature)

    # Remove diagonal elements
    ind = torch.eye(labels.size(0)).bool().to(device=DEVICE)
    sim_ij.masked_fill(ind, 0)
    exp_sim_matrix.masked_fill(ind, 0)

    # Compute the loss
    sum_exp_sim_matrix = torch.sum(exp_sim_matrix, dim=1)
    pos_exp_sim_matrix = torch.sum(sim_ij * positive_mask, dim=1)
    loss = -torch.log(pos_exp_sim_matrix / sum_exp_sim_matrix)

    return torch.mean(loss)


# def euler_rotation(theta, axis):
#     mat = Constant.IDENTITY.copy()

#     rot = rotation_from_angles(theta, axis)

#     mat[:3, :3] = rot

#     return mat


def AbsAngle_XZ1(srcs, dests):
    sx, sz = srcs[:, 0], srcs[:, 2]
    dx, dz = dests[:, 0], dests[:, 2]
    rad_a = np.arctan2(dx * sz - dz * sx, dx * sx + dz * sz)

    return np.abs(np.rad2deg(rad_a))


def AbsAngle_XZ2(src, dest):
    sx, sz = src[0], src[2]
    dx, dz = dest[0], dest[2]
    rad_a = np.arctan2(dx * sz - dz * sx, dx * sx + dz * sz)

    return np.abs(np.rad2deg(rad_a))


def SignedAngle_XZ(src, dest):
    sx, sz = src[0], src[2]
    dx, dz = dest[0], dest[2]
    rad_a = np.arctan2(dx * sz - dz * sx, dx * sx + dz * sz)

    return rad_a


def interpolate(fr, to, amount):
    return (1. - amount) * fr + amount * to


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0.:
        return vec
    else:
        return vec / norm


def lerp(v1, v2, t):
    ret = v1 + (v2 - v1) * t

    return ret


def look_rotation(forward, up):
    norm_f = np.linalg.norm(forward)
    if norm_f == 0:
        _forw = np.array([0., 0., 1.])
    else:
        _forw = forward / norm_f  # 앞 벡터
    _side = normalize(np.cross(up, _forw))  # 옆 벡터
    _up = normalize(np.cross(_forw, _side))  # 위 벡터

    return np.array([[_side[0], _up[0], _forw[0]],
                     [_side[1], _up[1], _forw[1]],
                     [_side[2], _up[2], _forw[2]]])


# up vector를 (0,1,0)으로, forward를 (x,0,z)일 때 더 빠른 look rotation
def look_rotation_XZ(forward):
    forw = normalize(forward)
    up = np.array([0., 1., 0.])
    side = normalize(np.cross(up, forw))

    return np.array([[side[0], up[0], forw[0]],
                     [side[1], up[1], forw[1]],
                     [side[2], up[2], forw[2]]])


def get_position_to(trs, to_inv):
    pos = trs[:, 3]
    return (to_inv @ pos)[:3]


def get_forward_to(trs, to_inv):
    dir = normalize(trs[:3, 2])
    dir = np.append(dir, [0])
    return (to_inv @ dir)[:3]


def get_up_to(trs, to_inv):
    dir = normalize(trs[:3, 1])
    dir = np.append(dir, [0])
    return (to_inv @ dir)[:3]


def get_velocity_to(vel, to_inv):
    return (to_inv @ vel)[:3]


def get_trs_to(trs, to_inv):
    return (to_inv @ trs)


# From mgen, rotation_matrix_3d.py https://github.com/NOhs/mgen
def _generate_matrix_XZX(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c2, -c3 * s2, s2 * s3],
         [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
         [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3]])


def _generate_matrix_XYX(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c2, s2 * s3, c3 * s2],
         [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
         [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3]])


def _generate_matrix_YXY(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
         [s2 * s3, c2, -c3 * s2],
         [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3]])


def _generate_matrix_YZY(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
         [c3 * s2, c2, s2 * s3],
         [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3]])


def _generate_matrix_ZYZ(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
         [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
         [-c3 * s2, s2 * s3, c2]])


def _generate_matrix_ZXZ(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
         [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
         [s2 * s3, c3 * s2, c2]])


def _generate_matrix_XZY(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c2 * c3, -s2, c2 * s3],
         [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
         [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3]])


def _generate_matrix_XYZ(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c2 * c3, -c2 * s3, s2],
         [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
         [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])


def _generate_matrix_YXZ(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
         [c2 * s3, c2 * c3, -s2],
         [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2]])


def _generate_matrix_YZX(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
         [s2, c2 * c3, -c2 * s3],
         [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3]])


def _generate_matrix_ZYX(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
         [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
         [-s2, c2 * s3, c2 * c3]])


def _generate_matrix_ZXY(c1, c2, c3, s1, s2, s3):
    return np.asarray(
        [[c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
         [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
         [-c2 * s3, s2, c2 * c3]])


def rotation_from_angles(theta, rotation_sequence):
    c1, c2, c3 = np.cos(theta)
    s1, s2, s3 = np.sin(theta)

    return globals()['_generate_matrix_' + rotation_sequence](c1, c2, c3, s1, s2, s3)


### Util Function for Training

def Normalize(data, axis, savefile=None):
    mean, std = data.mean(axis=axis), data.std(axis=axis)

    idx = np.where(std == 0.)
    std[idx] = 1.
    data = (data - mean) / std
    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')

    return data


def init_weight(rng, shape):
    weight_bound = np.sqrt(6. / np.prod(shape[-2:]))
    weight = np.asarray(
        rng.uniform(low=-weight_bound, high=weight_bound, size=shape),
        dtype=np.float32)
    return torch.from_numpy(weight)


def init_bias(shape):
    return torch.zeros(shape)

def compute_KL_div(mu, logvar, iteration):
    """ Compute KL divergence loss
        mu = (B, embed_dim)
        logvar = (B, embed_dim)
    """
    # compute KL divergence
    # see Appendix B from VAE paper:
    # D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", ICLR, 2014.

    kl_weight_center = 5000         # 중심 시점을 훨씬 앞당김
    kl_weight_growth_rate = 0.0005    # 증가 속도도 높임
    kl_threshold = 0.05

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B, )
    kl_div = torch.mean(kl_div)

    # compute weight for KL cost annealing:
    # S.R. Bowman, L. Vilnis, O. Vinyals, A.M. Dai, R. Jozefowicz, S. Bengio,
    # "Generating Sentences from a Continuous Space", arXiv:1511.06349, 2016.
    kl_div_weight = generalized_logistic_function(
        iteration, center=kl_weight_center, B=kl_weight_growth_rate,
    )
    # apply weight threshold
    kl_div_weight = min(kl_div_weight, kl_threshold)
    return kl_div, kl_div_weight


def generalized_logistic_function(x, center=0.0, B=1.0, A=0.0, K=1.0, C=1.0, Q=1.0, nu=1.0):
    """ Equation of the generalised logistic function
        https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param x:           abscissa point where logistic function needs to be evaluated
    :param center:      abscissa point corresponding to starting time
    :param B:           growth rate
    :param A:           lower asymptote
    :param K:           upper asymptote when C=1.
    :param C:           change upper asymptote value
    :param Q:           related to value at starting time abscissa point
    :param nu:          affects near which asymptote maximum growth occurs

    :return: value of logistic function at abscissa point
    """
    value = A + (K - A) / (C + Q * np.exp(-B * (x - center))) ** (1 / nu)
    return value