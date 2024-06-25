from dataclasses import dataclass
from enum import Enum


class WeightedType(Enum):
    class_type = "class"
    soft_type = "soft"


@dataclass
class Configure:
    NUM_CLASSES = 5
    IMG_SIZE = 448  # 训练时resize 注意:不推荐
    CROP_SIZE = 224  # 图片增强时random crop
    GRID_SIZE = CROP_SIZE  # 将高分辨图像切块分别进行处理
    PADDING = GRID_SIZE  # crop或者切块之后剩下的小图片进行padding
    FLIP_PROB = 0.4  # augmentation时flip概率
    BATCH_SIZE = 40
    NUM_EPOCHES = 100
    LEARNING_RATE = 0.001
    STEP_SIZE = 3  # 一些scheduler更新时的步数 比如steplr
    GAMMA = 0.1  # 一些scheduler的参数
    INTERVAL = 5  # 保存模型和打印损失等的epoch间隔
    NUM_WORKERS = 0  # dataloader的处理线程个数
    WEIGHTED = False  # nn.CrossEntropy的损失
    WEIGHTED_TYPE = WeightedType.class_type
    EXCLUDE_AREA = 1800  # mask后处理时面积
    MAX_OBJ_NUM = 5  # 每个物体最多的数量
    ENABLE_HALF = False  # 推理时使用半精度
    THIRD_PARTY_MODEL = True
    PLAIN_LOSS = True
