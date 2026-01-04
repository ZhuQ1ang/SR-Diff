
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from PIL import Image
from BLIPFeatureExtractor import BLIPFeatureExtractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_image_paths(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


if __name__ == "__main__":
    target_folder = input("请输入图片所在文件夹路径: ").strip()
    save_path = input("请输入CLS特征保存文件夹路径: ").strip()

    os.makedirs(save_path, exist_ok=True)


    images = get_image_paths(target_folder)
    print(f"共找到 {len(images)} 张图片。")

    blipmodel = BLIPFeatureExtractor()

    # 遍历处理
    for idx, path in enumerate(images, 1):
        image_ = Image.open(path).convert("RGB")
        filename = os.path.splitext(os.path.basename(path))[0]
        save_name = f"{filename}.pt"  
        cls_embedding = blipmodel.get_cls_embedding(image_)
        full_save_path = os.path.join(save_path, save_name)
        torch.save(cls_embedding, full_save_path)

        print(f"[{idx}/{len(images)}] 已保存: {full_save_path}")

