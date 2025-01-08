import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO




#### prepare data
rgb_file = '/data/guest_storage/zhanpengluo/Dataset/DAVIS/JPEGImages/480p/rollerblade/00000.jpg'

# intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
intrinsic = [489.01218, 435.0171, 191.5, 95.5]

extrinsic = torch.eye(4)
rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]


byte_stream = BytesIO()
byte_stream.seek(0)
image_bytes = byte_stream.getvalue()
new_byte_stream = BytesIO(image_bytes)

new_image = Image.open(new_byte_stream)
new_image.save("byte.jpg")

image = Image.fromarray(np.uint8(rgb_origin))
image.save('output.jpg')
# rgb_origin = cv2.resize(rgb_origin, (320, 180), interpolation=cv2.INTER_LINEAR)
# print(rgb_origin.shape)
# print("Above is our h w ")


# file_path = '/data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/dataset/re10k_subset/test/000000.torch'  # Replace with your file's path
# contents :list  = torch.load(file_path)


# print(f"length of list {len(contents)}")

# content = contents[0]

# for key, value in content.items():
#     if torch.is_tensor(value):
#         print(f"Key: {key}, Tensor shape: {value.shape}")
#     else:
#         print(f"Key: {key}, content type:", type(value))
        

# item_images = content["images"]
# print(len(item_images))

# print(item_images[0].shape)
