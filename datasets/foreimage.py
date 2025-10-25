import cv2
import numpy as np
import os

# # 输入和输出文件夹路径
# input_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/market_crops'
# mask_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/market_crops_label'
# output_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/cropped_market_objectgrounds/'
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 遍历输入文件夹中的所有图像文件
# for filename in os.listdir(input_folder):
#     if filename.endswith('.jpg'):  # 只处理 JPG 图像文件
#         # 构建图像和掩码的文件路径
#         image_path = os.path.join(input_folder, filename)
#         mask_path = os.path.join(mask_folder, filename.replace('.jpg', '.png'))  # 假设掩码文件名与图像相同，扩展名为 .png
#
#         # 读取图像和前景掩码
#         image = cv2.imread(image_path)  # 读取原始图像
#         image_resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
#
#         # 读取前景掩码，灰度模式
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
#         # 假设前景区域是非黑色的（即不为0的区域），我们可以创建掩码
#         foreground_mask = cv2.inRange(mask, 1, 255)  # 非黑色区域（前景）
#
#         # 获取前景区域的边界（最小矩形区域）
#         contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # 选择最大轮廓（假设它是我们需要的前景区域）
#         if contours:  # 确保有轮廓
#             largest_contour = max(contours, key=cv2.contourArea)
#
#             # 计算前景区域的边界框
#             x, y, w, h = cv2.boundingRect(largest_contour)
#
#             # 裁剪前景图像
#             foreground_cropped = image_resized[y:y + h, x:x + w]
#
#             # 构建保存路径
#             output_path = os.path.join(output_folder, f"{filename.replace('.jpg', '')}_foreground_cropped.png")
#
#             # 保存裁剪后的前景图像
#             cv2.imwrite(output_path, foreground_cropped)
#
#             print(f"Processed {filename} and saved cropped foreground to {output_path}")
#         else:
#             print(f"No contours found for {filename}, skipping...")
# #单一图像
# import cv2
# import numpy as np
#
# # 读取原始图像和前景掩码
# image = cv2.imread('/opt/data/private/yfz/PADE-main-single-3090/data/case/0155_c1_f0082036.jpg')  # 读取原始图像
# image_resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
#
# # 读取前景掩码，灰度模式
# mask = cv2.imread('/opt/data/private/yfz/PADE-main-single-3090/data/casem-label/0155_c1_f0082036.png', cv2.IMREAD_GRAYSCALE)
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
# # 使用掩码提取前景
# foreground_only = cv2.bitwise_and(image_resized, image_resized, mask=foreground_mask)
#
# # 获取前景区域的轮廓
# contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 选择最大的轮廓（假设它是我们需要的前景区域）
# largest_contour = max(contours, key=cv2.contourArea)
#
# # 计算前景区域的最小矩形框
# x, y, w, h = cv2.boundingRect(largest_contour)
#
# # 裁剪出前景图像
# foreground_cropped = image_resized[y:y+h, x:x+w]
#
# # 创建一个带透明度的图像，初始化为透明背景
# # 创建一个4通道图像（RGBA），其中前景部分的透明度为255，背景透明度为0
# foreground_with_transparency = np.dstack([foreground_cropped, np.zeros_like(foreground_cropped[:, :, 0])])
#
# # 将前景的 alpha 通道设置为掩码
# foreground_with_transparency[:, :, 3] = cv2.resize(foreground_mask[y:y+h, x:x+w], (foreground_cropped.shape[1], foreground_cropped.shape[0]))
#
# # 保存裁剪并带透明背景的前景图像
# cv2.imwrite('/opt/data/private/yfz/PADE-main-single-3090/data/case/foreground_cropped_with_transparency.png', foreground_with_transparency)
#
# print("Foreground cropped to the bounding box and saved with transparent background.")

# import cv2
# import numpy as np
# import os
#
# # 输入和输出文件夹路径
# input_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/occluded_crops'
# mask_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/occluded_crops_label'
# output_folder = '/opt/data/private/yfz/PADE-main-single-3090/data/cropped_occluded_foregrounds/'
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 遍历输入文件夹中的所有图像文件
# for filename in os.listdir(input_folder):
#     if filename.endswith('.jpg'):  # 只处理 JPG 图像文件
#         # 构建图像和掩码的文件路径
#         image_path = os.path.join(input_folder, filename)
#         mask_path = os.path.join(mask_folder, filename.replace('.jpg', '.png'))  # 假设掩码文件名与图像相同，扩展名为 .png
#
#         # 读取图像和前景掩码
#         image = cv2.imread(image_path)  # 读取原始图像
#         image_resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
#
#         # 读取前景掩码，灰度模式
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
#         # 打印分割图像的唯一值，帮助我们了解它的分布
#         print(f"Processing {filename}: Segmentation mask unique values: {np.unique(mask)}")
#
#         # 假设前景区域是非黑色的（即不为0的区域），我们可以创建掩码
#         foreground_mask = cv2.inRange(mask, 1, 255)  # 非黑色区域（前景）
#
#         # 确保掩码的大小与原图一致
#         foreground_mask = cv2.resize(foreground_mask, (image_resized.shape[1], image_resized.shape[0]))
#
#         # 确保掩码是 8 位单通道图像
#         foreground_mask = np.uint8(foreground_mask)
#
#         # 使用掩码提取前景
#         foreground_only = cv2.bitwise_and(image_resized, image_resized, mask=foreground_mask)
#
#         # 获取前景区域的轮廓
#         contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # 选择最大的轮廓（假设它是我们需要的前景区域）
#         largest_contour = max(contours, key=cv2.contourArea)
#
#         # 计算前景区域的最小矩形框
#         x, y, w, h = cv2.boundingRect(largest_contour)
#
#         # 裁剪出前景图像
#         foreground_cropped = image_resized[y:y + h, x:x + w]
#
#         # 创建一个带透明度的图像，初始化为透明背景
#         # 创建一个4通道图像（RGBA），其中前景部分的透明度为255，背景透明度为0
#         foreground_with_transparency = np.dstack([foreground_cropped, np.zeros_like(foreground_cropped[:, :, 0])])
#
#         # 将前景的 alpha 通道设置为掩码
#         foreground_with_transparency[:, :, 3] = cv2.resize(foreground_mask[y:y + h, x:x + w],
#                                                            (foreground_cropped.shape[1], foreground_cropped.shape[0]))
#
#         # 保存裁剪并带透明背景的前景图像
#         output_path = os.path.join(output_folder,
#                                    f"{filename.replace('.jpg', '')}_foreground_cropped_with_transparency.png")
#         cv2.imwrite(output_path, foreground_with_transparency)
#
#         print(f"Processed {filename} and saved cropped foreground to {output_path}")