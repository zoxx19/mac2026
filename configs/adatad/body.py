_base_ = [
    "../../_base_/datasets/mma52/e2e_train_trunc_test_sw.py",
    "../../_base_/models/actionformer.py",
]

window_size = 512
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 512/16=32 chunks

annotation_path = "./data/MMA-52/Annotations/adatad/mma52_anno.json"

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        data_path="data/body_crops/train",
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", suffix="_body"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        ann_file=annotation_path,
        data_path="data/body_crops/val",
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", suffix="_body"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        ann_file=annotation_path,
        data_path="data/body_crops/val",
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", suffix="_body"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/videomae-base/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,
            freeze_backbone=False,
        ),
    ),
    projection=dict(
        in_channels=768,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
    rpn_head=dict(
        num_classes=52,
        label_smoothing=0.1,
        loss=dict(
            cls_loss=dict(type="FocalLoss", weight=[0.4709, 0.6539, 1.0071, 3.8743, 0.7718, 0.4651, 0.6051, 0.3343, 0.2236, 0.2423, 0.2853, 0.5675, 0.5378, 0.8578, 1.4857, 0.9841, 1.2131, 0.4486, 1.0584, 0.3503, 0.7283, 0.3486, 0.5159, 0.4817, 0.3608, 0.7283, 0.8375, 0.8495, 0.9397, 0.5563, 0.5118, 0.5645, 1.1844, 1.2131, 1.4147, 1.7327, 1.8064, 0.9937, 1.0392, 1.19, 1.3368, 0.9209, 1.2914, 0.826, 1.2914, 4.6307, 2.0709, 1.2131, 0.4614, 0.3479, 0.8599, 2.3154]),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    train=dict(batch_size=8, num_workers=2),
    val=dict(batch_size=4, num_workers=2),
    test=dict(batch_size=4, num_workers=2),
    clip_grad_norm=0.5,
    amp=False,
    fp16_compress=False,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=5e-5,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=30)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=0,
    end_epoch=30,
)

work_dir = "work_dirs/e2e_mma52_body_videomae_b_adapter"
