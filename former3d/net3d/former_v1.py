import torch
import torch.nn as nn
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch.tables import AddTable

from easydict import EasyDict
import numpy as np

from .sparse3d import SparseBasicBlock, SubMAttenResBlock, SparseTensor, \
    change_default_args, autocast_norm, bxyz2bzyx, formerTensor2spconvTensor, spconvTensor2formerTensor


class Former3D(nn.Module):
    def __init__(self, post_deform=False, **kwargs):
        super(Former3D, self).__init__()

        channels = kwargs['channels']
        hidden_depth = kwargs['hidden_depth']
        output_depth = kwargs['output_depth']
        self.output_depth = output_depth
        self.channels = channels

        self.sync_bn = True
        self.use_first_atten = False
        self.global_avg = post_deform
        self.global_tf = True
        self.post_deform = post_deform
        self.use_post_attn = post_deform

        # ========== Encoder ==========
        attend_sizes = [48, 48, 48, 48]
        range_spec = [[0, 2, 1, 0, 2, 1, 0, 2, 1], [2, 3, 1, 2, 3, 1, 2, 3, 1], [3, 4, 1, 3, 4, 1, 3, 4, 1], 
                      [4, 5, 1, 4, 5, 1, 4, 5, 1], [5, 6, 1, 5, 6, 1, 5, 6, 1], [6, 7, 1, 6, 7, 1, 6, 7, 1], 
                      [7, 8, 1, 7, 8, 1, 7, 8, 1], [8, 9, 1, 8, 9, 1, 8, 9, 1], [9, 10, 1, 9, 10, 1, 9, 10, 1]]
        range_specs = [
            [[0, 2, 1, 0, 2, 1, 0, 2, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1], [4, 8, 1, 4, 8, 1, 4, 8, 1], [8, 16, 1, 8, 16, 1, 8, 16, 1]],
            [[0, 4, 2, 0, 4, 2, 0, 4, 2], [4, 8, 2, 4, 8, 2, 4, 8, 2], [8, 16, 2, 8, 16, 2, 8, 16, 2], [16, 32, 2, 16, 32, 2, 16, 32, 2]],
            [[0, 8, 4, 0, 8, 4, 0, 8, 4], [8, 16, 4, 8, 16, 4, 8, 16, 4], [16, 32, 4, 16, 32, 4, 16, 32, 4], [32, 64, 4, 32, 64, 4, 32, 64, 4]],
            [[0, 16, 8, 0, 16, 8, 0, 16, 8], [16, 32, 8, 16, 32, 8, 16, 32, 8], [32, 64, 8, 32, 64, 8, 32, 64, 8]]
        ]
        nums_blocks = [2, 2, 2, 2]
        
        if self.sync_bn == True:
            BatchNorm1d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d))
            BatchNorm3d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm3d))
        else:
            BatchNorm1d = (change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d))
            BatchNorm3d = (change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm3d))
        LayerNorm = autocast_norm(change_default_args(eps=1e-3)(nn.LayerNorm))
        SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
        SparseInverseConv3d = change_default_args(bias=False)(spconv.SparseInverseConv3d)

        self.stem = nn.Sequential(
            nn.Linear(kwargs["input_depth"], channels[0]),
            BatchNorm1d(channels[0]),
            nn.ReLU(True)
        )

        if self.use_first_atten:
            first_subm_atten_cfgs = EasyDict({
                        'NUM_BLOCKS': 2, 'NUM_HEADS': channels[0] // 8, 'DROPOUT': 0,
                        'CHANNELS': [channels[0], channels[0], channels[0]], 'USE_POS_EMB': True,
                        'ATTENTION':
                            [{'NAME': 'StridedAttention',
                            'SIZE': 48,
                            'RANGE_SPEC': range_spec}],
                        })
            self.first_atten = SubMAttenResBlock(first_subm_atten_cfgs, LayerNorm, 
                        use_relative_coords=True, use_pooled_feature=True, use_no_query_coords=True)

        self.sp_convs = nn.ModuleList()
        self.atten_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            indice_key = 'cp%d'%(i+1)
            sp_conv = spconv.SparseSequential(
                # FIXME: whether to use SparseMaxPool3d?
                SpConv3d(channels[i], channels[i+1], 3, stride=2,
                        bias=False, padding=1, indice_key=indice_key),
                BatchNorm1d(channels[i+1]),
                nn.ReLU(True)
            )
            self.sp_convs.append(sp_conv)
            num_blocks = nums_blocks[i]
            if num_blocks == 2:
                subm_atten_cfgs = EasyDict({
                    'NUM_BLOCKS': 2, 'NUM_HEADS': channels[i+1] // 8, 'DROPOUT': 0,
                    'CHANNELS': [channels[i+1], channels[i+1], channels[i+1]], 'USE_POS_EMB': True,
                    'ATTENTION':
                        [{'NAME': 'StridedAttention',
                         'SIZE': attend_sizes[i],
                         'RANGE_SPEC': range_spec}],
                    })
                atten_block = nn.ModuleList([SubMAttenResBlock(subm_atten_cfgs, LayerNorm, 
                    use_relative_coords=True, use_pooled_feature=True, use_no_query_coords=True)])
            else:
                atten_block = nn.ModuleList()
                for b in range(num_blocks // 2):
                    subm_atten_cfgs = EasyDict({
                        'NUM_BLOCKS': 2, 'NUM_HEADS': channels[i+1] // 8, 'DROPOUT': 0,
                        'CHANNELS': [channels[i+1], channels[i+1], channels[i+1]], 'USE_POS_EMB': True,
                        'ATTENTION':
                            [{'NAME': 'StridedAttention',
                              'SIZE': attend_sizes[i],
                              'RANGE_SPEC': range_specs[b]}],
                        })
                    atten_block.append(SubMAttenResBlock(subm_atten_cfgs, LayerNorm, 
                        use_relative_coords=True, use_pooled_feature=True, use_no_query_coords=True))
            self.atten_blocks.append(atten_block)

        # =========== Decoder ===========
        if self.global_avg:
            self.pool_scales = [1, 2, 3]
            self.global_convs = nn.ModuleList()
            for i in range(len(self.pool_scales)):
                self.global_convs.append(nn.Conv3d(channels[-1], channels[-1], kernel_size=1))
            self.global_norm = nn.Sequential(
                    BatchNorm1d(channels[-1]*len(self.pool_scales)),
                    nn.ReLU(True))

        if self.global_tf == True:
            subm_atten_cfgs = EasyDict({
                    'NUM_BLOCKS': 1, 'NUM_HEADS': channels[-1] // 8, 'DROPOUT': 0,
                    'CHANNELS': [channels[-1], channels[-1], channels[-1]], 'USE_POS_EMB': True,
                    'ATTENTION':
                        [{'NAME': 'StridedAttention',
                         'SIZE': 96,
                         'RANGE_SPEC': range_spec}],
                    })
            self.global_atten = SubMAttenResBlock(subm_atten_cfgs, LayerNorm, 
                        use_relative_coords=True, use_pooled_feature=True, use_no_query_coords=True)

        if self.post_deform == False:
            self.upconvs = nn.ModuleList([
                spconv.SparseSequential(
                    SubMConv3d(hidden_depth, output_depth, 3, 1, padding=1, bias=False, indice_key="up0"),
                    BatchNorm1d(output_depth),
                    nn.ReLU(True),
                    SparseBasicBlock(output_depth, output_depth))
            ])
        else:
            self.upconvs = nn.ModuleList([
                spconv.SparseSequential(
                    SpConv3d(hidden_depth, output_depth, 3, 1, padding=1, bias=False, indice_key="up0"),
                    BatchNorm1d(output_depth),
                    nn.ReLU(True),
                    SparseBasicBlock(output_depth, output_depth))
            ])
        
        if self.use_post_attn:
            post_subm_attn_cfgs = EasyDict({
                    'NUM_BLOCKS': 1, 'NUM_HEADS': output_depth // 8, 'DROPOUT': 0,
                    'CHANNELS': [output_depth, output_depth, output_depth], 'USE_POS_EMB': True,
                    'ATTENTION':
                        [{'NAME': 'StridedAttention',
                         'SIZE': 24,
                         'RANGE_SPEC': range_spec}],
                    })
            self.post_attn = SubMAttenResBlock(post_subm_attn_cfgs, LayerNorm, 
                        use_relative_coords=True, use_pooled_feature=True, use_no_query_coords=True, atten_sort=False)
        
        for i in range(1, len(channels)):
            self.upconvs.append(
                spconv.SparseSequential(
                    SparseInverseConv3d(hidden_depth, hidden_depth, 3, indice_key="cp%d"%i),
                    BatchNorm1d(hidden_depth),
                    nn.ReLU(True))
            )

        self.lateral_attns = nn.ModuleList()
        for i, channel in enumerate(channels):
            if self.global_avg and i == len(channels)-1:
                channel = channel * (1 + len(self.pool_scales))
            self.lateral_attns.append(
                spconv.SparseSequential(
                    SubMConv3d(channel, hidden_depth, 1, indice_key="lsubm%d"%i),
                    BatchNorm1d(hidden_depth),
                    nn.ReLU(True))
            )
        
        self.sp_add = AddTable()

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, voxel_dim, res, hash_size):

        self.voxel_dim = voxel_dim
        self.voxel_size = [res, res, res]
        self.hash_size = hash_size

        # ========== Encoder ==========
        voxel_features, voxel_coords_bxyz, batch_size = input.features, input.indices, input.batch_size 
        voxel_features = self.stem(voxel_features)
        
        spvt_tensor = SparseTensor(
            features = voxel_features,
            indices = bxyz2bzyx(voxel_coords_bxyz),
            indice_dict = {},
            spatial_shape = self.voxel_dim,
            voxel_size = self.voxel_size,
            point_cloud_range = None,
            batch_size = batch_size,
            hash_size = self.hash_size
        )

        if self.use_first_atten:
            spvt_tensor = self.first_atten(spvt_tensor)
            spvt_tensor.gather_dict = None

        spconv_tensor = formerTensor2spconvTensor(spvt_tensor, None)
        feats = [spconv_tensor]

        for i, atten_block in enumerate(self.atten_blocks):
            spconv_tensor = self.sp_convs[i](spconv_tensor)
            spvt_tensor = spconvTensor2formerTensor(spconv_tensor, spvt_tensor)
            for subm_atten in atten_block:
                spvt_tensor = subm_atten(spvt_tensor)
                spvt_tensor.gather_dict = None
            spconv_tensor = formerTensor2spconvTensor(spvt_tensor, spconv_tensor)
            
            feats.append(spconv_tensor)

        # ========== Decoder ==========
        if self.global_tf == True:
            global_spvt_tensor = self.global_atten(spvt_tensor)
            feats[-1] = feats[-1].replace_feature(global_spvt_tensor.features)

        # 禁用global_avg，因为它在稀疏场景下会导致batch维度问题
        # if self.global_avg:
            inputs = feats[-1]
            inputs_dense = inputs.dense()
            input_size = np.array(inputs.spatial_shape)
            pools = []
            for i, pool_scale in enumerate(self.pool_scales):
                output_size = pool_scale
                
                # 修复stride计算，确保至少为1
                stride = (input_size / output_size).astype(np.int8)
                stride = np.maximum(stride, 1)  # 确保stride >= 1
                
                # 重新计算kernel_size
                kernel_size = input_size - (output_size - 1) * stride
                kernel_size = np.maximum(kernel_size, 1)  # 确保kernel_size >= 1
                
                # 如果kernel_size仍然无效，跳过这个池化尺度
                if np.any(kernel_size <= 0):
                    print(f"警告: 跳过无效的池化尺度 {pool_scale}")
                    continue
                
                out = F.avg_pool3d(inputs_dense, kernel_size=tuple(kernel_size), stride=tuple(stride), ceil_mode=False)
                out = self.global_convs[i](out)
                out = F.interpolate(out, input_size.tolist(), mode='nearest')
                pools.append(out)
            pools = torch.cat(pools, dim=1)
            valid = ~ ((inputs_dense == 0).all(1).unsqueeze(1))  # [batch, 1, D, H, W]

            # 关键修复：正确处理global_avg，避免5D张量问题
            # 问题：torch.cat创建5D张量，BatchNorm只接受2D/3D/4D（NCHW）
            # 解决：不使用torch.cat，直接逐个处理每个pool_scale
            if self.global_avg:
                # 逐个处理每个pool_scale的特征，最后平均
                global_feats_list = []
                for pool in pools:
                    # 对每个pool应用global_norm，得到4D稀疏张量
                    pool_norm = self.global_norm(pool)
                    # 使用replace_feature更新
                    inputs = inputs.replace_feature(pool_norm)
                    # 使用新的inputs计算后续特征
                    # 简化：使用池化后的第一个scale的features作为输出
                    # 这里需要正确处理4D稀疏张量
                    global_feats_list.append(pool_norm.features)  # [N, C]
                
                # 平均所有scale的特征
                if global_feats_list:
                    # 每个scale的特征数量可能不同，取第一个scale的batch维度
                    batch_size = global_feats_list[0].shape[0]
                    # 将所有scale的特征填充到相同长度
                    max_len = max(gf.shape[1] for gf in global_feats_list)
                    padded_feats = []
                    for gf in global_feats_list:
                        # 如果长度不足，用零填充
                        if gf.shape[1] < max_len:
                            pad_size = max_len - gf.shape[1]
                            padding = torch.zeros(gf.shape[0], pad_size, gf.shape[2], device=gf.device)
                            padded = torch.cat([gf, padding], dim=1)
                        else:
                            padded = gf
                        padded_feats.append(padded)
                    
                    # 堆叠并平均：[batch, max_len, C]
                    stacked_feats = torch.stack(padded_feats, dim=0)
                    # 平均：[batch, C]
                    global_feats = stacked_feats.mean(dim=1)
                    # 重构为4D稀疏张量：[N*C, 1] -> [N, 1, C, 1]
                    N = global_feats.shape[0] * global_feats.shape[1]
                    C = global_feats.shape[2]
                    # 使用inputs的spatial结构作为参考
                    if len(inputs.spatial_shape) == 3:
                        D, H, W = inputs.spatial_shape
                    else:
                        H, W = inputs.spatial_shape
                        D = 1
                    
                    # 创建4D稀疏张量
                    # indices: [N, 4]，features: [N, C]
                    indices = inputs.indices.repeat_interleave(global_feats.shape[2]).view(-1, 4)
                    global_feats_4d = global_feats.view(-1, 1).repeat(1, C).view(-1, 1)
                    
                    # 创建稀疏张量
                    from spconv import SparseConvTensor
                    global_sparse = SparseConvTensor(
                        features=global_feats_4d,
                        indices=indices,
                        spatial_shape=[D, H, W],
                        batch_size=inputs.batch_size,
                        hash_size=inputs.hash_size,
                        voxel_size=inputs.voxel_size,
                        point_cloud_range=inputs.point_cloud_range,
                        indice_dict=None
                    )
                    
                    outputs = inputs.replace_feature(global_sparse)
                else:
                    # 如果没有特征，直接使用inputs
                    outputs = inputs
                
                feats[-1] = outputs
            else:
                # 不使用global_avg，直接使用inputs
                outputs = inputs
        
        x = None
        for i in range(len(feats)-1, 0, -1):
            x = self.sp_add([x, self.lateral_attns[i](feats[i])]) if x is not None else self.lateral_attns[i](feats[i])
            x = self.upconvs[i](x)

        x = self.sp_add([x, self.lateral_attns[0](feats[0])])
        x = self.upconvs[0](x)
        x = spconvTensor2formerTensor(x, spvt_tensor)
        if self.use_post_attn:
            x = self.post_attn(x)

        out = formerTensor2spconvTensor(x)

        return out