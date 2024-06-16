_base_ = ['./r50_nuimg_704x256_8f.py']

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    by_epoch=True,
    step=[48, 60],
    gamma=0.2
)
total_epochs = 60

# evaluation
eval_config = dict(interval=total_epochs)