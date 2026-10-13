import torch
from pointnet2_ops import pointnet2_utils as native

class ExportFPS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, count):
        return native.furthest_point_sample(xyz.contiguous(), count)

    @staticmethod
    def symbolic(g, xyz, count):
        result=g.op('pointr::FarthestPointSampling', xyz, count_i=count,
                    plugin_version_s='1', plugin_namespace_s='pointr')
        shape=xyz.type().sizes()
        return result.setType(xyz.type().with_dtype(torch.int32).with_sizes([shape[0], count]))

class pointnet2_utils:
    @staticmethod
    def furthest_point_sample(x, count):
        return ExportFPS.apply(x, count)

    @staticmethod
    def gather_operation(x, indices):
        return torch.gather(x, 2, indices.long().unsqueeze(1).expand(-1, x.shape[1], -1)).contiguous()
