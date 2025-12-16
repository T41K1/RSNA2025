# RSNA2025 2.5D Model

RSNA 2025 コンペのコード置き場です。3D ボリュームを 2D スライスとして扱う 2.5D 構成を採用し、timm バックボーンでスライスごとに特徴抽出 → LSTM でスライス方向を集約 → 残差付き 4 層 MLP で多ラベル分類を行います。

## リポジトリ構成
- `rsna2025_npy_train_033_2_5d(backbone_lstm_mlp(4layers+Residual),accums=1)_5epoch_dataset_6.ipynb`: すべてのコードと実験ログを含むノートブック

## モデル構成
- **バックボーン**: `tf_efficientnetv2_s.in21k_ft_in1k`（timm, pretrained, global average pooling, 出力 1280 次元）
- **2.5D パイプライン**: `D×H×W` の `.npy` ボリュームを `[B, T, C, H, W]` に正規化し、各スライスを 2D CNN に通して 1280 次元の列ベクトルに変換
- **時系列集約**: LSTM hidden 512, 1 layer, bidirectional=True（出力 1024 次元）、pad 対応、スライス方向は mean pooling（`agg="mean"`）
- **ヘッド (MLPHead)**: LayerNorm → Linear(1024→512) → GELU → Dropout(0.2) → Linear(512→512) で residual add → GELU → Dropout → Linear(512→256) → GELU → Dropout → Linear(256→14)
- **損失/評価**: `BCEWithLogitsLoss`、Weighted Macro-AUC（末尾クラス `Aneurysm Present` に weight 13、他は 1）

## データ
- `train.csv`（デフォルト: `/content/drive/MyDrive/DataScience/Kaggle/RSNA2025/train.csv`）と、シリーズ単位の `.npy` ボリューム群（デフォルト: `/content/preprocessed_6`）を前提
- `.npy` は Kaggle 公開データセットの `preprocessed_6` ディレクトリを複数バージョン（例: versions 2, 3）からシンボリックリンクで統合
- `RSNADataset` は存在する `.npy` のみ採用し、欠損は自動スキップ。`LABEL_COLS` は 14 クラス（左右 ICA/MCA/ACA/PComA、AComA、Basilar Tip、Other Posterior、Aneurysm Present）
- Albumentations による 2D オーグメンテーション（Resize 384、ShiftScaleRotate/Affine、Elastic/Grid/Optical Distortion、Gamma/BrightnessContrast、Noise/Blur、CoarseDropout 等）

## 学習設定（デフォルト）
- 入力サイズ 384, `in_chans=1`
- Stratified 5-fold（`Aneurysm Present` を基に層化）
- エポック 5、`batch_size=8`（train）、`valid_batch_size=2`、`grad_accum_steps=16`（実効バッチ 128 相当）
- Optimizer: AdamW をパラメータ群で分割  
  - backbone lr=1e-5, LSTM lr=1e-3, head lr=1e-3, `weight_decay=0.05`  
  - CosineAnnealingLR は `config.cosine=True` で有効化
- AMP 有効（`GradScaler` 使用）
- ベスト AUC 重み付きモデルを fold ごとに保存（`{out_dir}/{model_name}_fold{f}_best.pth`）

## 使い方のメモ
1. ノートブックを開き、`Config` 内の `csv` と `preprocessed_dir` を環境に合わせて設定
2. 必要ならデータ統合作成セル（`SRCS` → `DST` への symlink/copy）を実行して `.npy` を `/content/preprocessed_6` に集約
3. `main()` を実行すると 5-fold 学習と検証が走り、ログは `/content/drive/.../RSNA2025_output/logs` に保存されます（セル内の `_Tee` で標準出力をファイルへ複写）
