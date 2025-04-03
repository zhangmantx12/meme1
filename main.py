from utils.image_util import Insert
from utils.sign import MyYOLO
from utils.text_util import Image2Text
from pathlib import Path
import json
import os
from tqdm import tqdm


def get_all_images(dir):
    # 获取文件夹下所有 .jpg 文件
    jpg_files = [Path(f) for f in os.listdir(dir) if f.endswith('.jpg')]
    return jpg_files


def record_entry(json_file, entry):
    """
    向 JSON 文件中的数组添加一个条目，如果文件不存在则创建。
    :param json_file: JSON 文件路径
    :param entry: 要添加的条目（可以是字典或其他 JSON 支持的数据类型）
    """
    # 如果文件不存在，则初始化一个空数组
    if not os.path.exists(json_file):
        data = []
    else:
        # 如果文件存在，读取现有内容
        with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 确保数据是一个数组
                if not isinstance(data, list):
                    raise ValueError("JSON 文件内容不是数组")
                # 文件存在但内容无效，重新初始化为空数组
                data = []

    # 添加新的条目到数组
    data.append(entry)

    # 将更新后的数组写回文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"错误信息已添加到 {json_file}")


if __name__ == '__main__':
    yolo_path = "yolov8n.pt"
    api_key = "f456136e-9993-4bad-81c1-92520363d142"
    font_file = "assets/Arial Unicode.ttf"
    image_dir = "image"
    out_dir = Path("output_image_red")
    os.makedirs(out_dir,exist_ok=True)
    # init model
    yolo = MyYOLO(model_path=yolo_path)
    font_size = 25
    color = (255, 0, 0)  # black
    i2t = Image2Text(api_key=api_key)
    insert = Insert(font_file=font_file, size=font_size)

    ff = get_all_images(image_dir)
    # print(ff[0])
    for image_path in tqdm(ff):
        # image_path = Path("image/image_279.jpg")
        image_name = image_path.name

        image_path = image_dir / image_path
        # width, height = insert.get_image_message(image_path=image_path)
        x1, y1, x2, y2 = yolo.get_positon(image_path=image_path)
        out_path = out_dir / image_name
        text = i2t.get_image_text(image_path=image_path)
        x = (x1 + x2) / 2
        x = x - font_size * len(text) / 2
        y = max(font_size, y1 - font_size)
        positon = (x, y)
        insert.insert(text=text, image_path=image_path, out_path=out_path, position=positon, color=color)

        record_entry("error.json", str(image_path))
