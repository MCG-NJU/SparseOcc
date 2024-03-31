import torch 


def normalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]

    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos()], dim=-1)

    return out


def denormalize_bbox(normalized_bboxes):
    rot_sin = normalized_bboxes[..., 6:7]
    rot_cos = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sin, rot_cos)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        out = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    return out


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:7]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    if bboxes.shape[-1] > 7:
        vel = bboxes[..., 7:9].clone()
        return torch.cat([xyz, wlh, rot.sin(), rot.cos(), vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot.sin(), rot.cos()], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    if bboxes.shape[-1] > 8:
        vel = bboxes[..., 8:10].clone()
        return torch.cat([xyz, wlh, rot, vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot], dim=-1)

def bbox2occrange(bboxes, occ_size, query_cube_size=None):
    """
    xyz in [0, 1]
    wlh in [0, 1]
    """
    xyz = bboxes[..., 0:3].clone()
    if query_cube_size is not None:
        wlh = torch.zeros_like(xyz)
        wlh[..., 0] = query_cube_size[0]
        wlh[..., 1] = query_cube_size[1]
        wlh[..., 2] = query_cube_size[2]
    else:
        wlh = bboxes[..., 3:6]
        wlh[..., 0] = wlh[..., 0] * occ_size[0]
        wlh[..., 1] = wlh[..., 1] * occ_size[1]
        wlh[..., 2] = wlh[..., 2] * occ_size[2]
    
    xyz[..., 0] = xyz[..., 0] * occ_size[0]
    xyz[..., 1] = xyz[..., 1] * occ_size[1]
    xyz[..., 2] = xyz[..., 2] * occ_size[2]
    
    xyz = torch.round(xyz)
        
    low_bound = torch.round(xyz - wlh/2)
    high_bound = torch.round(xyz + wlh/2)
    
    return torch.cat((low_bound, high_bound), dim=-1).long()

def occrange2bbox(occ_range, occ_size, pc_range):
    """
    Return: xyz in [0, 1], wlh in [0, pc_range_size)
    """
    xyz = (occ_range[..., :3] + occ_range[..., 3:]).to(torch.float32) / 2
    xyz[..., 0] /= occ_size[0]
    xyz[..., 1] /= occ_size[1]
    xyz[..., 2] /= occ_size[2]
    wlh = (occ_range[..., 3:] - occ_range[..., :3]).to(torch.float32)
    wlh[..., 0] *= (pc_range[3] - pc_range[0]) / occ_size[0]
    wlh[..., 1] *= (pc_range[4] - pc_range[1]) / occ_size[1]
    wlh[..., 2] *= (pc_range[5] - pc_range[2]) / occ_size[2]
    return torch.cat((xyz, wlh), dim=-1)