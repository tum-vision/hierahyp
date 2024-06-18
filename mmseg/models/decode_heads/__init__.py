from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .sep_aspp_contrast_head import DepthwiseSeparableASPPContrastHead
from .ocr_head import OCRHead

__all__ = [
    'FCNHead', 'ASPPHead',
    'DepthwiseSeparableASPPContrastHead',
    'OCRHead'
]
