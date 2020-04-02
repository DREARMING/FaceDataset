import os
import shutil
import cv2
import numpy as np
import imghdr
from PIL import Image
from mtcnn.mtcnn import MTCNN
import face_recognition

MIN_FACE_SIZE = 40
FACE_MARGIN = 20
FACE_PIC_SIZE = 160
FACE_TOLERANCE = 0.6
# 明星图片数据集最少数量阈值
MIN_PIC_NUM = 3

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
        if img.shape[0] < FACE_PIC_SIZE or img.shape[1] < FACE_PIC_SIZE:
            print("图片太小了：%d - %d" % (img.shape[0], img.shape[1]))
            return False
        return True
    except:
        return False


# size[0] == height, size[1] == width
def extends_face_margin(box, size):
    box[0] = max(box[0] - FACE_MARGIN, 0)
    box[1] = max(box[1] - FACE_MARGIN, 0)
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


def makeValidDir(index):
    dir = str(index)
    while len(dir) < 8:
        dir = '0' + dir
    return dir


def resize_pic(box, img):
    width = box[2] - box[0]
    height = box[3] - box[1]
    # 先让宽高取最大值
    if width > height:
        height = width
    else:
        width = height
    if width < FACE_PIC_SIZE:
        height = width = FACE_PIC_SIZE
        return crop_cube_img(box, width, img)
    else:
        crop = crop_cube_img(box, width, img)
        resize_size = (FACE_PIC_SIZE, FACE_PIC_SIZE)
        minify_img = cv2.resize(crop, resize_size, interpolation=cv2.INTER_AREA)
        return minify_img


# 该函数的作用是，将 box 的宽高拓展到 size，并裁剪图片出来作为返回值
# 因为会涉及到边缘的情况，这里为了确保返回的图片是 size * size 的，会在边缘情况中心点会迁移
def crop_cube_img(box, size, img):
    new_box = [0, 0, 0, 0]
    x_extend = size - (box[2] - box[0])
    half = int(x_extend // 2)
    if half >= box[0]:
        new_box[0] = 0
        new_box[2] = size
    elif box[2] + half >= img.shape[1]:
        new_box[2] = img.shape[1]
        new_box[0] = new_box[2] - size
    else:
        new_box[0] = box[0] - half
        new_box[2] = new_box[0] + size
    y_extend = size - (box[2] - box[0])
    half = int(y_extend // 2)
    if half >= box[1]:
        new_box[1] = 0
        new_box[3] = size
    elif box[3] + half >= img.shape[0]:
        new_box[3] = img.shape[0]
        new_box[1] = new_box[3] - size
    else:
        new_box[1] = box[1] - half
        new_box[3] = new_box[1] + size
    return img[new_box[1]:new_box[3], new_box[0]:new_box[2]]


# 清除数据集中不属于同一个人的照片
def del_not_same_person(processed_path):
    try:
        class_paths = os.listdir(processed_path)
        for class_path in class_paths:
            class_dir = os.path.join(processed_path, class_path)
            img_paths = os.listdir(class_dir)
            embeddings = []
            for img_path in img_paths:
                pic_path = os.path.join(class_dir, img_path)
                # 用 face_recognition 来提取人脸特征码
                try:
                    img_data = face_recognition.load_image_file(pic_path)
                    embedding_result = face_recognition.face_encodings(img_data, num_jitters=10)
                    if len(embedding_result) == 0:
                        os.remove(pic_path)
                        print("提取人脸特征结果为空，移除该照片：%s" % pic_path)
                        continue
                    embeddings.append(embedding_result[0])
                except Exception as e:
                    os.remove(pic_path)
                    print("提取人脸特征失败，移除该照片：%s" % pic_path)
                    print(e)
                    continue
            # 重新获取文件列表，因为提取特征如果发生异常的话，index就乱了
            img_paths = os.listdir(class_dir)
            for i, embedding in enumerate(embeddings):
                results = face_recognition.compare_faces(embeddings, embedding, tolerance=FACE_TOLERANCE)
                # 用 np 将 True/False 数组转成一个整数
                sum = np.sum(np.array(results).astype(np.int64))
                # 如果该图片与其他图片被认为是一个人的数量不足半数，认为是一张不合法的图片，移除
                # 当图片数量 == 2 或者 == 1 时，这种算法是非常不合理的，那么我们可以在数据集中处理，图片少于3的都不要了
                if sum < int(len(embeddings) // 2):
                    os.remove(os.path.join(class_dir, img_paths[i]))
                    print("该图片不属于该数据集，删除：%s" % os.path.join(class_dir, img_paths[i]))
            img_paths = os.listdir(class_dir)
            # 目录空的话，就删除掉
            if len(img_paths) == 0:
                os.removedirs(class_dir)
    except Exception as e:
        print(e)
    pass


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
            index += 1
            new_name_path = makeValidDir(index)
            target_dir = os.path.join(processed_path, new_name_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for image_path in image_paths:
                # 获取图片路径
                img_path = os.path.join(father_path, name_path, image_path)
                valid = check_file(img_path)
                # 不合法的照片要删掉
                if not valid:
                    print("%s 的图片：%s 不合法" % (name_path, img_path))
                    os.remove(img_path)
                    continue
                result = detect_face(img_path)
                if not result:
                    os.remove(img_path)
                    continue
                box = result[0]
                img = result[1]
                face_box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                print("face box ; %s, shape : %s" % (str(face_box), str(img.shape)))
                # 拓展 margin 个像素，然后再裁剪
                extends_face_margin(face_box, img.shape)
                print("new box : %s" % str(face_box))
                # 裁剪出 FACE_PIC_SIZE * FACE_PIC_SIZE 大小的人脸图
                resize_img = resize_pic(face_box, img)
                # 裁剪 图片, crop 样式为 y0:y1 - x0:x1
                # cropped = img[face_box[1]:face_box[3], face_box[0]:face_box[2]]

                # 删除源文件，重写裁剪过的文件
                os.remove(img_path)

                # 将裁剪好的图片写入新的目录中
                new_img_path = os.path.join(target_dir, image_path)
                cv2.imwrite(new_img_path, resize_img)
            # 删除旧的目录
            shutil.rmtree(os.path.join(father_path, name_path))
            # 如果一个明星的图片数量太少，不用该明星的数据集
            file_list = os.listdir(target_dir)
            if len(file_list) < MIN_PIC_NUM:
                shutil.rmtree(target_dir)
        print('清理完成')
        # 清除不是同一个人的人脸
        del_not_same_person(processed_path)
    except Exception as e:
        print(e)
        pass