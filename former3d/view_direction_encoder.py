import numpy as np
import torch


class ViewDirectionEncoder(torch.nn.Module):
    def __init__(self, feat_depth, L):
        super().__init__()
        self.L = L
        self.view_embedding_dim = 3 + self.L * 6
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                feat_depth + self.view_embedding_dim, feat_depth, 1, bias=False
            ),
        )
        torch.nn.init.xavier_normal_(self.conv[0].weight)

    def forward(self, feats, proj, cam_positions_dict):
        device = feats.device
        dtype = feats.dtype
        featheight, featwidth = feats.shape[2:]
        
        # 获取当前分辨率的相机位置
        # 注意：cam_positions_dict是整个字典，我们需要从中提取
        # 这里假设调用者传递了正确的字典
        # 在实际调用中，应该是cam_positions[resname]
        
        # 简化：假设cam_positions_dict已经是正确分辨率的张量
        if isinstance(cam_positions_dict, dict):
            # 如果是字典，我们需要知道当前分辨率
            # 这里简化处理，使用第一个键
            cam_positions = list(cam_positions_dict.values())[0]
        else:
            cam_positions = cam_positions_dict
            
        u = torch.arange(featwidth, device=device)
        v = torch.arange(featheight, device=device)
        vv, uu = torch.meshgrid(v, u) #, indexing='ij')
        ones = torch.ones_like(uu)
        uv = torch.stack((uu, vv, ones, ones), dim=0).to(dtype)

        inv_proj = torch.linalg.inv(proj.to(dtype))
        xyz = inv_proj @ uv.reshape(4, -1)
        # 重塑xyz以匹配cam_positions的形状
        xyz_reshaped = xyz.reshape(inv_proj.shape[0], inv_proj.shape[1], 4, -1)
        view_vecs = xyz_reshaped[:, :, :3, :] - cam_positions.unsqueeze(-1)
        view_vecs /= torch.linalg.norm(view_vecs, dim=2, keepdim=True)


        # 重塑view_vecs以进行编码
        view_vecs_flat = view_vecs.reshape(view_vecs.shape[0], view_vecs.shape[1], 3, -1)
        view_vecs_flat = view_vecs_flat.permute(0, 1, 3, 2)  # [batch, n_imgs, H*W, 3]
        
        view_encoding = [view_vecs_flat]
        for i in range(self.L):
            view_encoding.append(torch.sin(view_vecs_flat * np.pi * 2 ** i))
            view_encoding.append(torch.cos(view_vecs_flat * np.pi * 2 ** i))
        view_encoding = torch.cat(view_encoding, dim=-1).to(dtype)  # [batch, n_imgs, H*W, embedding_dim]
        
        # 重塑回特征图形状
        view_encoding = view_encoding.reshape(
            view_encoding.shape[0] * view_encoding.shape[1],
            view_encoding.shape[3],
            featheight,
            featwidth,
        )

        feats = torch.cat((feats, view_encoding), dim=1)
        feats = self.conv(feats)
        return feats
