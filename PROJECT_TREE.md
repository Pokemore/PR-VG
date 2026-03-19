```
/data/1c7611/DOCUMENTS/
├── Code/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── main.py
│   │   └── README.md
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── categories.py
│   │   ├── coco_eval.py
│   │   ├── concat_dataset.py
│   │   ├── image_to_seq_augmenter.py
│   │   ├── refer.py
│   │   ├── refexp_eval.py
│   │   ├── refexp.py
│   │   ├── refexp2seq.py
│   │   ├── rsvg_hr.py
│   │   ├── rsvg_mm.py
│   │   ├── rsvg.py
│   │   ├── samplers.py
│   │   └── transforms_image.py
│   ├── docs/
│   ├── engine.py
│   ├── inference_rsvg1.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   ├── criterion.py
│   │   ├── deformable_transformer.py
│   │   ├── matcher.py
│   │   ├── ops/
│   │   │   ├── build/
│   │   │   │   └── lib.linux-x86_64-cpython-310/
│   │   │   │       └── MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so
│   │   │   ├── dist/
│   │   │   │   ├── MultiScaleDeformableAttention-1.0-py3.12-linux-x86_64.egg
│   │   │   │   ├── MultiScaleDeformableAttention-1.0-py3.7-linux-x86_64.egg
│   │   │   │   └── MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg
│   │   │   ├── functions/
│   │   │   │   ├── __init__.py
│   │   │   │   └── ms_deform_attn_func.py
│   │   │   ├── make.sh
│   │   │   ├── modules/
│   │   │   │   ├── __init__.py
│   │   │   │   └── ms_deform_attn.py
│   │   │   ├── MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so
│   │   │   ├── MultiScaleDeformableAttention.egg-info/
│   │   │   │   ├── dependency_links.txt
│   │   │   │   ├── PKG-INFO
│   │   │   │   ├── SOURCES.txt
│   │   │   │   └── top_level.txt
│   │   │   ├── setup.py
│   │   │   ├── src/
│   │   │   │   ├── cpu/
│   │   │   │   │   ├── ms_deform_attn_cpu.cpp
│   │   │   │   │   └── ms_deform_attn_cpu.h
│   │   │   │   ├── cuda/
│   │   │   │   │   ├── ms_deform_attn_cuda.cu
│   │   │   │   │   ├── ms_deform_attn_cuda.h
│   │   │   │   │   └── ms_deform_im2col_cuda.cuh
│   │   │   │   ├── ms_deform_attn.h
│   │   │   │   └── vision.cpp
│   │   │   └── test.py
│   │   ├── position_encoding.py
│   │   ├── postprocessors.py
│   │   ├── PR-VG.py
│   │   ├── segmentation.py
│   │   ├── sparse_conv_profiler.py
│   │   ├── sparse_conv.py
│   │   ├── swin_transformer.py
│   │   ├── text_guided_pruning.py
│   │   └── video_swin_transformer.py
│   ├── opts.py
│   ├── Overall.pdf
│   ├── Overall.png
│   ├── PR-VG.tex
│   ├── README.md
│   ├── Shell/
│   │   ├── Ablation/
│   │   │   ├── DIOR-RSVG/
│   │   │   │   ├── Pring+DVR/
│   │   │   │   │   ├── DIOR_ablation_test_Pruning+DVR.sh
│   │   │   │   │   └── DIOR_ablation_train_Pruning+DVR.sh
│   │   │   │   ├── Pruning/
│   │   │   │   │   ├── DIOR_ablation_test_Pruning.sh
│   │   │   │   │   └── DIOR_ablation_train_Pruning.sh
│   │   │   │   └── Pruning+IoU/
│   │   │   │       ├── DIOR_ablation_test_Pruning+IoU.sh
│   │   │   │       └── DIOR_ablation_train_Pruning+IoU.sh
│   │   │   └── OPT-RSVG/
│   │   │       ├── Pruning/
│   │   │       │   ├── OPT_ablation_test_Pruning.sh
│   │   │       │   └── OPT_ablation_train_Pruning.sh
│   │   │       ├── Pruning+DVR/
│   │   │       │   ├── OPT_ablation_test_Pruning+DVR.sh
│   │   │       │   └── OPT_ablation_train_Pruning+DVR.sh
│   │   │       └── Pruning+IoU/
│   │   │           ├── OPT_ablation_test_Pruning+IoU.sh
│   │   │           └── OPT_ablation_train_Pruning+IoU.sh
│   │   └── Result/
│   │       ├── DIOR-RSVG/
│   │       │   ├── PR-VG/
│   │       │   │   ├── PR-VG_test.sh
│   │       │   │   └── PR-VG_train.sh
│   │       │   ├── PR-VG-B/
│   │       │   │   ├── PR-VG-B_test.sh
│   │       │   │   └── PR-VG-B_train.sh
│   │       │   └── PR-VG-L/
│   │       │       ├── PR-VG-L_test.sh
│   │       │       └── PR-VG-L_train.sh
│   │       └── OPT-RSVG/
│   │           ├── PR-VG/
│   │           │   ├── PR-VG_test.sh
│   │           │   └── PR-VG_train.sh
│   │           ├── PR-VG-C/
│   │           │   ├── PR-VG-C_test.sh
│   │           │   └── PR-VG-C_train.sh
│   │           └── PR-VG-E/
│   │               ├── PR-VG-E_test.sh
│   │               └── PR-VG-E_train.sh
│   ├── tools/
│   │   ├── colormap.py
│   │   ├── data/
│   │   │   ├── convert_davis_to_ytvos.py
│   │   │   └── convert_refexp_to_coco.py
│   │   ├── load_pretrained_weights.py
│   │   └── verify_importance_vectorization.py
│   ├── torch_patch.py
│   ├── train_logs/
│   │   ├── none_0t2vw860/
│   │   │   └── attempt_0/
│   │   │       ├── 0/
│   │   │       ├── 1/
│   │   │       └── 2/
│   │   └── none_qgdz0jnm/
│   │       └── attempt_0/
│   │           ├── 0/
│   │           ├── 1/
│   │           └── 2/
│   ├── util/
│   │   ├── __init__.py
│   │   ├── box_ops.py
│   │   ├── misc.py
│   │   └── transforms.py
│   └── Ziyang_Wang-6.pdf
├── Dataset/
│   ├── DIOR_RSVG/
│   │   ├── Annotations/
│   │   ├── JPEGImages/
│   │   ├── dataset.txt
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
│   └── OPT-RSVG/
│       ├── Annotations/
│       ├── Annotations.zip
│       ├── Image.zip
│       ├── JPEGImages/
│       ├── test.txt
│       ├── train.txt
│       └── val.txt
├── Model/
│   ├── Ablation/
│   │   ├── DIOR-RSVG/
│   │   │   ├── IoU/
│   │   │   │   └── checkpoint.pth
│   │   │   ├── Pruning/
│   │   │   │   └── checkpoint0064.pth
│   │   │   ├── Pruning+DVR/
│   │   │   │   └── checkpoint0064.pth
│   │   │   └── Pruning+IoU/
│   │   │       └── checkpoint0049.pth
│   │   └── OPT-RSVG/
│   │       ├── IoU/
│   │       │   └── checkpoint.pth
│   │       ├── Pruning/
│   │       │   └── checkpoint0069.pth
│   │       ├── Pruning+DVR/
│   │       │   └── checkpoint0069.pth
│   │       └── Pruning+IoU/
│   │           └── checkpoint0049.pth
│   ├── Result/
│   │   ├── DIOR-RSVG/
│   │   │   ├── PR-VG/
│   │   │   │   └── checkpoint0049.pth
│   │   │   ├── PR-VG-B/
│   │   │   │   └── checkpoint0049.pth
│   │   │   └── PR-VG-L/
│   │   │       └── checkpoint0069.pth
│   │   └── OPT-RSVG/
│   │       ├── PR-VG/
│   │       │   └── checkpoint0069.pth
│   │       ├── PR-VG-C/
│   │       │   └── checkpoint0069.pth
│   │       └── PR-VG-E/
│   │           └── checkpoint0054.pth
│   └── sensitivity Analysis/
│       ├── 0.6-0.9.pth
│       ├── 0.75-0.75.pth
│       └── 0.82-0.72.pth
└── Pretrain/
    └── RoBERTa-base/
        ├── config.json
        ├── dict.txt
        ├── flax_model.msgpack
        ├── merges.txt
        ├── model.safetensors
        ├── pytorch_model.bin
        ├── README.md
        ├── rust_model.ot
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json
```
