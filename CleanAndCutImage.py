import os
import shutil
import cv2
import numpy as np
import imghdr
from PIL import Image
from mtcnn.mtcnn import MTCNN

MIN_FACE_SIZE = 40
FACE_MARGIN = 20

# 设置 min_face_size == 40 pixes
detector = MTCNN(min_face_size=MIN_FACE_SIZE)

def check_file(img_path):
    try:
        # 获取图片的类型
        image_type = imghdr.what(img_path)
        # 如果图片格式不是JPEG同时也不是PNG就删除图片
        if image_type != 'jpeg' and image_type != 'png':
            return False
        # 删除灰度图
        img = np.array(Image.open(img_path))
        if len(img.shape) is 2:
            return False
        return True
    except:
        return False


# size[0] == height, size[1] == width
def extends_face_margin(box, size):
    box[0] = max(box[0] - FACE_MARGIN, 0)
    box[1] = max(box[0] - FACE_MARGIN, 0)
    box[2] = min(box[2] + FACE_MARGIN, size[1])
    box[3] = min(box[3] + FACE_MARGIN, size[0])


def detect_face(img_path):
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        result = detector.detect_faces(img)
        num_of_face = len(result)
        if num_of_face != 1:
            print("人脸数不为1, 删除该照片：%s" % img_path)
            return None
        if result[0]['confidence'] < 0.85:
            print("人脸可信度低于 0.85, remove : %s" % img_path)
            return None
        return result[0]['box'], img
    except:
        pass
    return None

# 3. 预先加载图片，如果图片加载失败，或者不是jpg或者png的抛弃掉。然后用 mtcnn扫描图片，
#    如果没有人脸或者人脸数大于1，抛弃掉。 mtcnn返回bounding_boxes, 可以在这里，对图片进行裁剪。
# 4. 对个人的照片集进行提取特征码操作，用face_reg 来操作
# 1. 先进行一次遍历，把特征码都提取了。
# 2. 然后遍历特征码列表，进行 compare 操作，和其他照片相似的数量小于图片总数量一定比例的，抛弃掉，可能不是这个明星的照片，混合进去的。


if __name__ == '__main__':
    father_path = 'star_image'
    processed_path = 'star_image_processed'
    try:
        name_paths = os.listdir(father_path)
        index = 0
        for name_path in name_paths:
            print('正在清理 %s 的图片...' % name_path)
            image_paths = os.listdir(os.path.join(father_path, name_path))
            size = len(image_paths)
            for image_path in image_paths:
                # 获取图片路径
                img_path = os.path.join(father_path, name_path, image_path)
                valid = check_file(img_path)
                # 不合法的照片要删掉
                if not valid:
                    print("%s 的图片：%s 不合法" % (name_path, img_path))
                    os.remove(img_path)
                    size -= 1
                    continue
                box, img = detect_face(img_path)
                if not box:
                    size -= 1
                    os.remove(img_path)
                    print('box is null')
                    continue
                face_box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                print("face box ; %s, shape : %s" % (str(face_box), str(img.shape)))
                # 拓展 margin 个像素，然后再裁剪
                extends_face_margin(face_box, img.shape)
                print("new box : %s" % str(face_box))
                # resize 图片, crop 样式为 y0:y1 - x0:x1
                cropped = img[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                # 删除源文件，重写裁剪过的文件
                os.remove(img_path)
                cv2.imwrite(img_path, cropped)
                size -= 1
            if size:
                shutil.move(src=os.path.join(father_path, name_path), dst=os.path.join(processed_path, name_path))
        print('清理完成')
    except Exception as e:
        print(e)
        pass