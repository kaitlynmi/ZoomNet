_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True
deterministic = True
use_custom_worker_init = False
model_name = "ZoomNet"

train = dict(
    batch_size=4,
    num_workers=4,
    use_amp=True,
    num_epochs=200,
    epoch_based=True,
    lr=0.001,
    optimizer=dict(
        mode="sgd",
        set_to_none=True,
        group_mode="finetune",
        cfg=dict(
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="f3",
        cfg=dict(
            lr_decay=0.9,
            min_coef=0.001,
        ),
    ),
)

val = dict(
    batch_size=4,
    num_workers=2,
    interval = 1,
    show_bar=False,
    tta=dict(
        enable=False,
    ),
    clip_range=None,
)

test = dict(
    batch_size=1,
    num_workers=2,
    show_bar=False,
    evaluate=False,
)

datasets = dict(
    train=dict(
        dataset_type="carotid_cod_tr",
        shape=dict(h=384, w=384),
        path=["carotid_train"],
        interp_cfg=dict(),
    ),
    val=dict(
        dataset_type="carotid_cod_val",
        shape=dict(h=384, w=384),
        path=["carotid_val"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="carotid_cod_te",
        shape=dict(h=384, w=384),
        path=["carotid_val"],
        interp_cfg=dict(),
    ),
    # test=dict(
    #     dataset_type="carotid_cod_te_img",
    #     shape=dict(h=384, w=384),  # Resize to 384x384
    #     path=["carotid_ak_84"],
    #     interp_cfg=dict(),
    # ),
)
