_base_ = [
    '../configs/_base_/datasets/coco_detection.py', '../configs/_base_/default_runtime.py'
]


##############
# 1. 训练设定
# 16
train_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during training
# 8
train_num_workers = 1
# persistent_workers must be False if num_workers is 0
persistent_workers = True
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
max_epochs = 100  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10
# load_from = './model/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938-ce3c1b3f.pth'
img_scale = (1280, 1280)  # width, height
val_batch_size_per_gpu = 1
val_num_workers = 2

##############
# 1. 数据集设定
dataset_type = 'CocoDataset'
classes = ('with_rips', )
data_root = 'E:/Data/Research/Rip/video/' 

train_ann_file =  data_root + 'annotations/trainval.json'
train_data_prefix = data_root + 'images/'  
val_ann_file =  data_root + 'annotations/trainval.json'
val_data_prefix =  data_root + 'images/' 
test_ann_file =  data_root + 'annotations/test.json'
test_data_prefix =  data_root + 'images/' 

num_classes = 1  
metainfo = dict(classes=classes, palette=[(220, 20, 60)])

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_data_prefix)
        )
    )
test_evaluator = dict(
    ann_file=test_ann_file,
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

val_evaluator = dict(
    ann_file=val_ann_file,
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

# data = dict(
#     train=dict(
#         img_prefix=train_data_prefix,
#         classes=classes,
#         ann_file=train_ann_file),
#     val=dict(
#         img_prefix=val_data_prefix,
#         classes=classes,
#         ann_file=val_ann_file),
#     test=dict(
#         img_prefix=test_data_prefix,
#         classes=classes,
#         ann_file=test_ann_file))

model = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

# train_dataloader = dict(
#     # dataset=dict(
#     #     filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))
#     dataset=dict(
#         type=dataset_type,
#         metainfo=dict(classes=classes),
#         data_root=data_root,
#         ann_file=train_ann_file,
#         data_prefix=dict(img=train_data_prefix),
#         filter_cfg=dict(filter_empty_gt=False),
#         pipeline=train_pipeline))

# train_dataloader = train_dataloader
# val_dataloader = train_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 60
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

save_epoch_intervals = 1
max_keep_ckpts = 3
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))



visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend', init_kwargs=dict(name="Dino_4S_Rip", project='rip-currents'))],
    name='visualizer')