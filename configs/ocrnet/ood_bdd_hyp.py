#_base_ = './ocrnet_hr18_4xb2-80k_cityscapes-512x1024_origin.py'
_base_ = [
    '../_base_/models/ocrnet_hr18_origin.py', '../_base_/datasets/bdd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',gamma=0,use_focal=True, use_sigmoid=False, loss_weight=0.4)),
# model = dict(
#     pretrained='open-mmlab://msra/hrnetv2_w48',
#     backbone=dict(
#         extra=dict(
#             stage2=dict(num_channels=(48, 96)),
#             stage3=dict(num_channels=(48, 96, 192)),
#             stage4=dict(num_channels=(48, 96, 192, 384)))),
#     decode_head=[
#         dict(
#             type='FCNHead',
#             in_channels=[48, 96, 192, 384],
#             channels=sum([48, 96, 192, 384]),
#             input_transform='resize_concat',
#             in_index=(0, 1, 2, 3),
#             kernel_size=1,
#             num_convs=1,
#             norm_cfg=norm_cfg,
#             concat_input=False,
#             dropout_ratio=-1,
#             num_classes=26,
#             align_corners=False,
#             loss_decode=dict(
#                 type='HieraTripletLossCityscape_Euclidean',num_classes=19, use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            hyperbolic=True,
            loss_decode=dict(
                type='CrossEntropyLoss', gamma=0, use_focal=True, use_sigmoid=False, loss_weight=1))
    ],
    test_cfg=dict(mode='whole', is_hiera=False, hiera_num_classes=7))
