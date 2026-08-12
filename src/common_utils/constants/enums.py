from enum import Enum


class ImageTriggerType(str, Enum):
    BLENDED_PATCH = "blended_patch"
    CHECKERBOARD = "checkerboard"
    NOISE = "noise"


class ImageTriggerLocation(str, Enum):
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    CENTER = "center"


# This enum is used as the main switch in your PoisoningConfig
class PoisonType(str, Enum):
    NONE = "none"
    LABEL_FLIP = "label_flip"
    IMAGE_BACKDOOR = "image_backdoor"
    TEXT_BACKDOOR = "text_backdoor"
    TABULAR_BACKDOOR= "tabular_backdoor"


class ImageBackdoorAttackName(str, Enum):
    SIMPLE_DATA_POISON = "simple_data_poison"
    SIMPLE_GRADIENT_MANIPULATION = "simple_gradient_manipulation"


class TextBackdoorAttackName(str, Enum):
    SIMPLE_DATA_POISON = "simple_data_poison"


class TextTriggerLocation(str, Enum):
    START = "start"
    END = "end"
    MIDDLE = "middle"


class VictimStrategy(str, Enum):
    RANDOM = "random"
    FIXED = "fixed"


class LabelFlipMode(str, Enum):
    FIXED_TARGET = "fixed_target"
    RANDOM_DIFFERENT = "random_different"


class ServerAttackMode(str, Enum):
    NONE = "none"
    GRADIENT_INVERSION = "gradient_inversion"
