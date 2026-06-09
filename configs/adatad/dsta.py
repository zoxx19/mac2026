_base_ = [
    "../../_base_/datasets/mma52/e2e_train_trunc_test_sw.py",
    "../../_base_/models/actionformer.py",
]

# Dual-path Spatial-Temporal Adapter (DSTA) run.
# Warm-started (weights only) from the proven adapter ep22 checkpoint; the DSTA
# adapters are fresh (identity init) and learn from a short fresh warmup.
# NOTE: amp/fp16 are DISABLED on purpose -- fp16 produced Loss=inf on prior runs.

window_size = 512
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 512/16=32 chunks

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
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
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
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
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
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
            type="VisionTransformerDSTA",
            img_size=224,
            patch_size=16,
            embed_dims=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,  # activation checkpointing -> fits fp32
            total_frames=window_size * scale_factor,
            adapter_index=list(range(24)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/videomae_large.pth",
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
        in_channels=1024,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
    rpn_head=dict(
        num_classes=52,
        label_smoothing=0.1,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    # NOTE: batch_size is the GLOBAL batch (divided across GPUs) and must be
    # divisible by world_size. 4 GPUs -> train 8 = 2/GPU (ep22 used global 4).
    train=dict(batch_size=8, num_workers=8),
    val=dict(batch_size=4, num_workers=8),
    test=dict(batch_size=4, num_workers=8),
    clip_grad_norm=0.5,
    amp=False,  # fp16 caused Loss=inf previously -> train in fp32
    fp16_compress=False,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
# Short fresh warmup for the new DSTA adapters. Budget is ~4h (~2 epochs), so
# warmup_start_lr is 1e-5 (not 1e-6) to keep epoch 0 productive; epoch 1 ~= 1e-4.
# (warmup_epoch must be >= 2: the scheduler divides by warmup_epoch-1.)
scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=2,
    warmup_start_lr=1e-5,
    max_epoch=30,
)

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
    checkpoint_interval=1,  # save every epoch -- the time budget is tight
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=30,
)

work_dir = "work_dirs/e2e_mma52_videomae_l_dsta"
