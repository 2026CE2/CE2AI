# 3D Scene Prediction with Transformers

Exam mini-project for the *AI and Advanced Machine Learning* course  
(Module IX – Foundations of Generative Modeling II, Aalborg University).

Predicts a structured 3D scene representation (depth map, semantic segmentation, object 3D positions) from a single RGB image using a Vision Transformer (ViT) encoder–decoder.

---

## Theoretical background

### Transformer architecture (Vaswani et al., 2017)

The model is built on the **"Attention is All You Need"** paper. The key building blocks are:

#### 1. Scaled Dot-Product Attention

Each input vector plays three roles – **Query (Q)**, **Key (K)**, **Value (V)** – obtained via learnable linear projections:

$$A(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Dividing by $\sqrt{d_k}$ prevents the dot products from growing too large, which would push softmax into regions with very small gradients.

#### 2. Multi-Head Attention

Instead of a single attention computation in $d_m$ dimensions, $h$ heads each operate in $d_m / h$ dimensions, then concatenate and project back:

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)$$

In the original paper: $h = 8$, $d_k = d_v = 64$.

#### 3. Positional Encoding

Since all tokens are processed in parallel, positional information must be injected explicitly using sinusoidal encodings:

$$p[i](t) = \begin{cases} \sin(\omega_k \cdot t) & i = 2k \\ \cos(\omega_k \cdot t) & i = 2k+1 \end{cases}, \quad \omega_k = \frac{1}{10000^{2k/d_m}}$$

This allows the model to generalise to sequence lengths not seen during training without extra parameters.

#### 4. Encoder–Decoder architecture

| Component | Description |
|---|---|
| **Encoder** | N × (Multi-Head Self-Attention + FFN), each with residual + LayerNorm |
| **Decoder** | N × (Masked Self-Attention + Cross-Attention + FFN), each with residual + LayerNorm |
| **Output** | Linear projection → Softmax / regression heads |

#### 5. Vision Transformer (ViT) adaptation

Following Dosovitskiy et al. (2020), the image is split into fixed-size patches (16 × 16), each flattened and linearly projected to $d_m$ dimensional tokens. These tokens are the input sequence to the encoder.

---

## Project structure

```
mini-project/
├── configs/
│   └── config.yaml              # All hyperparameters
├── data/
│   ├── __init__.py
│   ├── dataset.py               # SceneDataset + DataLoader factory
│   └── preprocessing.py         # Normalisation & augmentation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # AbsRel, RMSE, mIoU, MED, …
│   └── visualize.py             # Training curves, attention maps
├── images/                      # Sample / reference images
├── models/
│   ├── __init__.py
│   ├── attention.py             # Scaled dot-product attention
│   ├── multi_head_attention.py  # Multi-head attention
│   ├── positional_encoding.py   # Sinusoidal & learnable PE
│   ├── feed_forward.py          # Position-wise FFN
│   ├── encoder.py               # EncoderLayer + TransformerEncoder
│   ├── decoder.py               # DecoderLayer + TransformerDecoder
│   ├── transformer.py           # Full encoder-decoder Transformer
│   └── scene_predictor.py       # ViT-based 3D scene predictor
├── training/
│   ├── __init__.py
│   ├── loss.py                  # SceneLoss (depth + semantic + position)
│   ├── scheduler.py             # Noam warm-up & cosine schedules
│   └── trainer.py               # Training loop, checkpointing, early stop
├── utils/
│   ├── __init__.py
│   └── helpers.py               # set_seed, get_device, count_parameters
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Data format

Place your dataset under `data/raw/` with this structure:

```
data/raw/
├── images/       ← RGB images  (*.jpg / *.png)
├── depth/        ← Depth maps  (*.npy, float32)
├── semantic/     ← Class masks (*.npy, int64)
└── poses/        ← Object poses (*.json)
```

Each JSON pose file follows:
```json
[
  {"position": [0.12, 0.45, 2.3], "class_id": 5},
  ...
]
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train
python main.py --config configs/config.yaml --mode train

# 3. Evaluate a saved checkpoint
python main.py --config configs/config.yaml --mode eval \
               --checkpoint checkpoints/checkpoint_best.pt
```

---

## Key hyperparameters (`configs/config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 512 | Embedding dimension |
| `num_heads` | 8 | Attention heads |
| `num_enc_layers` | 6 | Encoder depth |
| `num_dec_layers` | 6 | Decoder depth |
| `d_ff` | 2048 | FFN hidden size |
| `patch_size` | 16 | ViT patch size |
| `num_queries` | 100 | Decoder output tokens |
| `warmup_steps` | 4000 | Noam LR warm-up |

---

## References

1. Vaswani, A. et al. (2017). *Attention is All You Need*. NeurIPS 30.  
2. Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words*. arXiv:2010.11929.  
3. Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473.  
4. Raschka, S. & Mirjalili, V. (2022). *Python Machine Learning*, 3rd ed. Packt Publishing.  
5. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
