# import cv2
# import numpy as np
#
# # 读取原始图像和前景掩码
# image = cv2.imread('/opt/data/private/yfz/PADE-main-single-3090/data/object/0013_c6_f0081542.jpg')  # 读取原始图像
# image_resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
#
# # 读取前景掩码，灰度模式
# mask = cv2.imread('/opt/data/private/yfz/PADE-main-single-3090/data/object-label/0013_c6_f0081542.png', cv2.IMREAD_GRAYSCALE)
#
# # 打印掩码中的唯一值，帮助我们了解前景区域
# print(f"Segmentation mask unique values: {np.unique(mask)}")
#
# # 假设前景区域是非黑色的（即不为0的区域），我们可以创建掩码
# foreground_mask = cv2.inRange(mask, 1, 255)  # 非黑色区域（前景）
#
# # 确保掩码的大小与原图一致
# foreground_mask = cv2.resize(foreground_mask, (image_resized.shape[1], image_resized.shape[0]))
#
# # 确保掩码是 8 位单通道图像
# foreground_mask = np.uint8(foreground_mask)
#
# # 创建背景掩码（背景是掩码值为 0 的区域）
# background_mask = cv2.bitwise_not(foreground_mask)
#
# # 使用掩码提取背景（背景部分）
# background_only = cv2.bitwise_and(image_resized, image_resized, mask=background_mask)
#
# # 创建一个带透明度的图像，初始化为透明背景
# # 创建一个4通道图像（RGBA），其中背景部分的透明度为255，前景透明度为0
# background_with_transparency = np.dstack([background_only, np.zeros_like(background_only[:, :, 0])])
#
# # 将背景的 alpha 通道设置为掩码
# background_with_transparency[:, :, 3] = cv2.resize(background_mask, (background_only.shape[1], background_only.shape[0]))
#
# # 保存裁剪并带透明背景的背景图像
# cv2.imwrite('/opt/data/private/yfz/PADE-main-single-3090/data/object/background_with_transparency.png', background_with_transparency)
#
# print("Background extracted and saved with transparent background.")
# import cv2
# import numpy as np
# import random
#
# # 读取原始图像和前景掩码
# image = cv2.imread('/opt/data/private/yfz/PADE-main-single-3090/data/object/0013_c6_f0081542.jpg')
# image_resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
#
# # 读取前景掩码，灰度模式
# mask = cv2.imread('/opt/data/private/yfz/PADE-main-single-3090/data/object-label/0013_c6_f0081542.png',
#                   cv2.IMREAD_GRAYSCALE)
#
# # 打印掩码中的唯一值
# print(f"Segmentation mask unique values: {np.unique(mask)}")
#
# # 创建前景掩码（非黑色为前景）
# foreground_mask = cv2.inRange(mask, 1, 255)
# foreground_mask = cv2.resize(foreground_mask, (image_resized.shape[1], image_resized.shape[0]))
#
# # 创建背景掩码
# background_mask = cv2.bitwise_not(foreground_mask)
#
# # 获取背景区域坐标范围
# background_indices = np.where(background_mask != 0)
# min_y, min_x = np.min(background_indices[0]), np.min(background_indices[1])
# max_y, max_x = np.max(background_indices[0]), np.max(background_indices[1])
#
# # 四个角坐标
# corners = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]
#
# # 重试控制
# max_retries = 100
# retries = 0
# foreground_ratio_limit = 0.1  # 前景不超过 10%
#
# cropped_image = None
#
# while retries < max_retries:
#     corner = random.choice(corners)
#     corner_x, corner_y = corner
#
#     # 随机裁剪尺寸
#     crop_width = random.randint(50, max_x - min_x)
#     crop_height = random.randint(50, max_y - min_y)
#
#     # 根据角计算裁剪起点
#     if corner == (min_x, min_y):  # 左上
#         crop_x, crop_y = corner_x, corner_y
#     elif corner == (max_x, min_y):  # 右上
#         crop_x, crop_y = corner_x - crop_width, corner_y
#     elif corner == (min_x, max_y):  # 左下
#         crop_x, crop_y = corner_x, corner_y - crop_height
#     else:  # 右下
#         crop_x, crop_y = corner_x - crop_width, corner_y - crop_height
#
#     # 越界保护
#     if crop_x < 0 or crop_y < 0 or crop_x + crop_width > 128 or crop_y + crop_height > 256:
#         retries += 1
#         continue
#
#     # 裁剪图像和前景掩码
#     cropped_image = image_resized[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
#     cropped_fg_mask = foreground_mask[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
#
#     # 计算前景比例
#     fg_ratio = np.sum(cropped_fg_mask > 0) / (crop_width * crop_height)
#
#     # 若前景比例小于阈值，接受
#     if fg_ratio <= foreground_ratio_limit:
#         print(f"Found valid crop with foreground ratio: {fg_ratio:.3f}")
#         break
#
#     retries += 1
#
# if cropped_image is not None:
#     cv2.imwrite('/opt/data/private/yfz/PADE-main-single-3090/data/object/crop.png', cropped_image)
#     print("Cropped image saved.")
# else:
#     print("Warning: Unable to find a suitable crop with low enough foreground.")
import cv2
import numpy as np
import random
import os

# 输入文件夹和输出文件夹路径
input_image_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/occluded_crops'
input_mask_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/occluded_crops_label'
output_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/crops3_occluded_objectimage/'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_image_folder) if f.endswith('.jpg')]

# 重试控制
max_retries = 100
foreground_ratio_limit = 0.1  # 前景不超过 10%

# 遍历所有图像文件
for image_file in image_files:
    # 构建图像和掩码的文件路径
    image_path = os.path.join(input_image_folder, image_file)
    mask_path = os.path.join(input_mask_folder, image_file.replace('.jpg', '.png'))

    # 读取图像和掩码
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 打印掩码中的唯一值
    print(f"Processing image: {image_file}")
    print(f"Segmentation mask unique values: {np.unique(mask)}")

    # 创建前景掩码（非黑色为前景）
    foreground_mask = cv2.inRange(mask, 1, 255)
    foreground_mask = cv2.resize(foreground_mask, (image_resized.shape[1], image_resized.shape[0]))

    # 创建背景掩码
    background_mask = cv2.bitwise_not(foreground_mask)

    # 获取背景区域坐标范围
    background_indices = np.where(background_mask != 0)
    min_y, min_x = np.min(background_indices[0]), np.min(background_indices[1])
    max_y, max_x = np.max(background_indices[0]), np.max(background_indices[1])

    # 四个角坐标
    corners = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]

    retries = 0
    cropped_image = None

    while retries < max_retries:
        corner = random.choice(corners)
        corner_x, corner_y = corner

        # 随机裁剪尺寸
        crop_width = random.randint(50, max_x - min_x)
        crop_height = random.randint(50, max_y - min_y)

        # 根据角计算裁剪起点
        if corner == (min_x, min_y):  # 左上
            crop_x, crop_y = corner_x, corner_y
        elif corner == (max_x, min_y):  # 右上
            crop_x, crop_y = corner_x - crop_width, corner_y
        elif corner == (min_x, max_y):  # 左下
            crop_x, crop_y = corner_x, corner_y - crop_height
        else:  # 右下
            crop_x, crop_y = corner_x - crop_width, corner_y - crop_height

        # 越界保护
        if crop_x < 0 or crop_y < 0 or crop_x + crop_width > 128 or crop_y + crop_height > 256:
            retries += 1
            continue

        # 裁剪图像和前景掩码
        cropped_image = image_resized[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        cropped_fg_mask = foreground_mask[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        # 计算前景比例
        fg_ratio = np.sum(cropped_fg_mask > 0) / (crop_width * crop_height)

        # 若前景比例小于阈值，接受
        if fg_ratio <= foreground_ratio_limit:
            print(f"Found valid crop for {image_file} with foreground ratio: {fg_ratio:.3f}")
            break

        retries += 1

    if cropped_image is not None:
        # 保存裁剪后的图像
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_crop.png")
        cv2.imwrite(output_image_path, cropped_image)
        print(f"Cropped image saved: {output_image_path}")
    else:
        print(f"Warning: Unable to find a suitable crop for {image_file} with low enough foreground.")



