# RSNA 2025 — Intracranial Aneurysm Detection with 2.5D CNN-LSTM

Kaggle コンペティション [RSNA 2025 Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection) に取り組んだソリューションです。

脳血管の 3D CTA（CT Angiography）ボリュームから **脳動脈瘤（Aneurysm）の有無と発生部位（14 クラス）** を予測するマルチラベル分類タスクに対して、**2.5D CNN + Bidirectional LSTM + Residual MLP** アーキテクチャを設計・実装しました。

## コンペティション概要

| 項目 | 内容 |
|------|------|
| **主催** | Radiological Society of North America (RSNA) |
| **タスク** | 頭部 3D CTA から脳動脈瘤の有無・発生部位を予測（マルチラベル分類） |
| **データ** | DICOM 形式の 3D CTA ボリューム（元データ約 200 GB 超） |
| **評価指標** | Weighted Macro-AUC（Aneurysm Present に weight 13、他 13 クラスは weight 1） |
| **対象部位（14 クラス）** | 左右 ICA（Infraclinoid / Supraclinoid）、左右 MCA、AComA、左右 ACA、左右 PComA、Basilar Tip、Other Posterior Circulation、Aneurysm Present |

## アーキテクチャ

3D ボリュームを直接扱うのではなく、**スライスごとに 2D CNN で特徴抽出し、スライス方向を LSTM で集約する 2.5D アプローチ**を採用しました。

```
Input: 3D Volume (.npy)
  [D, H, W]
     │
     ▼
┌─────────────────────────────┐
│  _normalize_to_BTCHW        │  任意の入力形状を [B, T, C, H, W] に正規化
│  (3D/4D/5D → 統一形式)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  EfficientNetV2-S (timm)    │  各スライスを独立に 2D CNN で処理
│  pretrained on ImageNet-21k │  → 1280-dim feature per slice
│  Global Average Pooling     │
└─────────────┬───────────────┘
              │  [B, T, 1280]
              ▼
┌─────────────────────────────┐
│  Bidirectional LSTM         │  スライス方向の時系列的な依存関係を学習
│  hidden=512, layers=1       │  → 1024-dim (512×2) per timestep
│  pack_padded_sequence 対応   │
└─────────────┬───────────────┘
              │  [B, T, 1024]
              ▼
┌─────────────────────────────┐
│  Mean Pooling over T        │  可変長スライスを固定長に集約
│  (マスク付き)                │
└─────────────┬───────────────┘
              │  [B, 1024]
              ▼
┌─────────────────────────────┐
│  MLPHead (4-layer + Res.)   │  LayerNorm → FC(1024→512) → GELU
│                             │  → FC(512→512) + Residual → GELU
│                             │  → FC(512→256) → GELU
│                             │  → FC(256→14)
└─────────────┬───────────────┘
              │  [B, 14]
              ▼
         BCEWithLogitsLoss
```

### 設計上の工夫

- **2.5D 構成**: 3D CNN は計算コスト・メモリが大きいため、2D CNN + LSTM で擬似的に 3D 情報を扱う。各スライスの空間特徴を 2D CNN で抽出し、スライス間の連続性を LSTM で学習する
- **可変長入力への対応**: `pack_padded_sequence` を使用し、シリーズごとにスライス枚数が異なる場合にもパディング + マスク付き Mean Pooling で対応
- **Residual MLP Head**: 4 層の MLP ヘッドに Residual Connection を導入し、勾配消失を抑制しつつ表現力を確保
- **パラメータグループの分離学習率**: backbone（微調整: 1e-5）、LSTM（1e-3）、MLP Head（1e-3）で学習率を分離し、事前学習済み重みの破壊を防止

## 学習設定

| パラメータ | 値 |
|-----------|-----|
| **バックボーン** | `tf_efficientnetv2_s.in21k_ft_in1k` (timm, pretrained) |
| **入力サイズ** | 384 × 384, 1ch (grayscale) |
| **交差検証** | Stratified 5-Fold（`Aneurysm Present` で層化） |
| **エポック数** | 5 |
| **バッチサイズ** | train=8, valid=2, `grad_accum_steps=16`（実効バッチ 128） |
| **Optimizer** | AdamW（weight_decay=0.05） |
| **Scheduler** | CosineAnnealingLR（オプション） |
| **AMP** | 有効（GradScaler 使用） |
| **Data Augmentation** | ShiftScaleRotate / Affine / ElasticTransform / GridDistortion / OpticalDistortion / RandomGamma / BrightnessContrast / GaussNoise / MotionBlur / CoarseDropout 等 |

## データパイプライン

```
元データ (DICOM, ~200GB)
     │
     │  Kaggle 環境上で前処理
     ▼
.npy ボリューム (D×H×W, float)
     │
     │  複数バージョンを symlink で統合
     ▼
preprocessed_6/
  ├── {SeriesInstanceUID_1}.npy
  ├── {SeriesInstanceUID_2}.npy
  └── ...
     │
     │  RSNADataset: 存在する .npy のみ自動採用（欠損スキップ）
     ▼
DataLoader → [B, T, C, H, W] → Model
```

- 元の DICOM データが 200 GB 超と巨大であるため、Kaggle 環境上で事前に `.npy` に変換・圧縮した前処理済みデータを使用
- Kaggle 公開データセットの複数バージョンをシンボリックリンクで統合し、ストレージ使用量を最小化
- `RSNADataset` クラスは初期化時にファイル存在チェックを実行し、欠損データを自動でスキップする設計

## リポジトリ構成

```
RSNA2025/
├── README.md
└── rsna2025_npy_train_033_2_5d(...).ipynb   # 全コード・実験ログを含むノートブック
```

## 実行方法

1. ノートブックを Google Colab（GPU ランタイム）で開く
2. `Config` クラス内の `csv` と `preprocessed_dir` を環境に合わせて設定
3. 必要に応じてデータ統合セル（symlink 作成）を実行し、`.npy` を統合ディレクトリに集約
4. `main()` セルを実行 → 5-Fold 学習・検証が開始
5. 学習ログは `{OUTPUT_ROOT}/logs/` にファイル出力される（`_Tee` による標準出力の複写）

## 今後の課題 / 次に試したいこと

### 1. DICOM → npy 変換パイプラインの自前実装

今回は Kaggle 上で公開されている前処理済み `.npy` データセットを利用したが、元の DICOM から `.npy` への変換パイプラインを自前で構築するところでつまづいた。DICOM のメタデータ（Slice Thickness、Pixel Spacing、Rescale Slope/Intercept など）を正しくハンドリングしながらボリュームを再構成し、適切な Window レベルで正規化する部分に経験不足を感じた。データ前処理の段階をブラックボックスにせず自分でコントロールできるようになることが重要な課題。

### 2. ROI 抽出を行う 2 段階モデルの構築

現在のパイプラインではボリューム全体をそのままモデルに入力しているが、脳動脈瘤はボリューム全体のごく一部の領域に存在する。そのため、**第 1 段階で脳血管領域の ROI（Region of Interest）を検出・クロップし、第 2 段階でクロップされた領域に対して分類を行う 2 段階パイプライン**が有効だと考えられる。

ROI のクロップ精度が低いと後段の分類精度にも直接影響するため、Detection モデル（例: YOLO、Faster R-CNN）や Segmentation モデル（例: U-Net）で血管領域を正確に抽出する仕組みが必要。上位ソリューションの多くがこの 2 段階アプローチを採用しており、単一モデルでの精度改善には限界があると考えている。

### 3. その他の改善案

- **3D CNN / 3D Vision Transformer** の導入によるスライス間情報のより直接的な活用
- **Test Time Augmentation (TTA)** の適用
- **マルチスケール入力** による異なる解像度での特徴抽出
- **Pseudo Labeling** / Semi-Supervised Learning の活用

## 技術スタック

| カテゴリ | ライブラリ |
|---------|-----------|
| Deep Learning | PyTorch, timm |
| Data Augmentation | Albumentations |
| CV | scikit-learn (StratifiedKFold, roc_auc_score) |
| 画像処理 | OpenCV, pydicom, scipy |
| 実行環境 | Google Colab (GPU) |
