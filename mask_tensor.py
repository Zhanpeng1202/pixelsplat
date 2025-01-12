import torch
from torch import Tensor

# mask: Tensor = torch.tensor([[[[True, False, True],
#                                [False, True, False],
#                                [True, False, True]]]])

# print(mask.shape)


# b n 3 
pcd = torch.rand(1, 3, 3)
mask = torch.tensor([True, False, True]).unsqueeze(0)
print(mask.shape)
print(pcd.shape)


# # point_cloud: shape [b, 3, 3]
# # mask: shape [b, 1], containing True/False or 1/0

# First, make sure `mask` is a 1D boolean tensor of shape [b]
mask_1d = mask.squeeze(dim=-1).bool()

# Now index the point_cloud with mask_1d
filtered_point_cloud = pcd[mask_1d]
print(filtered_point_cloud.shape)
