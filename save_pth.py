import torch


# def print_pth(contents):
#     if isinstance(contents, list):

#         for content in contents:
#             print_pth(content)
#         return
    
#     elif isinstance(contents, dict):
#         for key, value in contents.items():
#             if torch.is_tensor(value):
#                 print(f"Key: {key}, Tensor shape: {value.shape}")
#             else:
#                 # print(f"Key: {key}, Value: {value}")
#                 print_pth(contents)
#                 return
#                 print("content type:", type(contents))
#     elif torch.is_tensor(contents):
#         print(contents.shape)
#     else:
#         print("Unknown content type:", type(contents))

file_path = '/data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/dataset/re10k_subset/test/000000.torch'  # Replace with your file's path
contents :list  = torch.load(file_path)


print(f"length of list {len(contents)}")

content = contents[0]

for key, value in content.items():
    if torch.is_tensor(value):
        print(f"Key: {key}, Tensor shape: {value.shape}")
    else:
        print(f"Key: {key}, content type:", type(value))
        

item_images = content["images"]

for i in item_images:
    print(f" content type:", type(i))
