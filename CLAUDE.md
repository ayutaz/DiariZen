# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

DiariZenは、[AudioZen](https://github.com/haoxiangsnr/spiking-fullsubnet)と[Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)をベースとした話者ダイアライゼーション(speaker diarization)ツールキットです。音声データから複数の話者を識別し、「誰がいつ話したか」を時系列で分類します。

### 主な特徴
- 自己教師あり学習(SSL)ベースのWavLMモデルを使用
- 構造化プルーニング(structured pruning)によるモデル軽量化に対応
- Hugging Face統合による簡単な推論実行
- 複数のベンチマークデータセットで高精度を達成

### ライセンス
- **コード**: MIT License
- **事前学習済みモデルの重み**: CC BY-NC 4.0 (研究・非商用利用のみ)

## 環境構築

### 必須要件
- Python 3.10
- CUDA 12.1対応のGPU (学習時)
- uv (Python パッケージマネージャー)

### インストール手順 (uv使用)

```bash
# 1. Python 3.10仮想環境の作成
uv venv --python 3.10

# 2. 仮想環境のアクティベート (PowerShell)
.venv\Scripts\Activate.ps1

# 3. PyTorchのインストール (CUDA 12.1)
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# 4. DiariZenの依存関係とパッケージのインストール
# pesqとpystoiはWindows環境でVisual Studio Build Toolsが必要なためスキップ
uv pip install einops flit h5py joblib jupyterlab tensorboard librosa matplotlib "numpy==1.26.4" onnxruntime-gpu openpyxl pandas pre-commit pyyaml scipy soundfile tabulate toml torchinfo tqdm "accelerate==1.6.0" thop
uv pip install -e .

# 5. pyannote-audioのインストール
uv pip install -e "pyannote-audio[dev,testing]"

# 6. dscoreサブモジュールの初期化
git submodule init
git submodule update
```

### モデルのダウンロード

事前学習済みモデルをHugging Faceからダウンロード：

```bash
# 仮想環境をアクティベートした状態で実行

# DiariZen Largeモデル（推奨・高精度）
python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='BUT-FIT/diarizen-wavlm-large-s80-md', cache_dir='./models')"

# DiariZen Baseモデル（軽量版）
python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='BUT-FIT/diarizen-wavlm-base-s80-md', cache_dir='./models')"

# 話者埋め込みモデル（必須）
python -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='pyannote/wespeaker-voxceleb-resnet34-LM', filename='pytorch_model.bin', cache_dir='./models')"
```

モデルは `./models/` ディレクトリに保存されます。

### 動作確認

環境構築後、サンプル音声で動作確認：

```bash
python test_inference.py
```

正常に動作すれば、話者分離結果が表示されます。

### 依存関係の注意点
- `accelerate==1.6.0`は互換性のため固定バージョン
- `numpy==1.26.4`も固定バージョン
- pyannote-audioは開発版として別途インストールが必要
- `pesq`と`pystoi`はVisual Studio Build Toolsが必要（音声品質評価用で、コア機能には不要）
- Windows環境では文字エンコーディングの問題を避けるため、スクリプト内で `sys.stdout.reconfigure(encoding='utf-8')` の設定を推奨

## よく使うコマンド

### 推論 (Hugging Face経由)

```python
from diarizen.pipelines.inference import DiariZenPipeline

# 事前学習済みモデルの読み込み
diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")

# ダイアライゼーションの実行
diar_results = diar_pipeline('./example/EN2002a_30s.wav')

# 結果の表示
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
```

### モデル学習

#### Single Optimizer (Fbank, WavLM-frozen)
```bash
cd recipes/diar_ssl
# run_stage.shの設定を編集:
# - use_dual_opt=false
# - train_conf を選択 (fbank_conformer.toml または wavlm_frozen_conformer.toml)

CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --num_processes 2 --main_process_port 1134 \
    run_single_opt.py -C conf/fbank_conformer.toml -M train
```

#### Dual Optimizer (WavLM-updated)
```bash
cd recipes/diar_ssl
# run_stage.shの設定を編集:
# - use_dual_opt=true
# - train_conf=conf/wavlm_updated_conformer.toml

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
    --num_processes 4 --main_process_port 1134 \
    run_dual_opt.py -C conf/wavlm_updated_conformer.toml -M train
```

### モデルプルーニング

```bash
cd recipes/diar_ssl_pruning
# run_stage.shを実行 (事前に設定を編集)
bash run_stage.sh
```

### 推論とスコアリング

```bash
cd recipes/diar_ssl

# 推論
python infer_avg.py -C <config_path> \
    -i <wav.scp> \
    -o <output_dir> \
    --embedding_model <embedding_model_path> \
    --avg_ckpt_num 5 \
    --val_metric Loss \
    --val_mode best \
    --seg_duration 8 \
    --clustering_method AgglomerativeClustering \
    --ahc_threshold 0.70 \
    --min_cluster_size 30

# スコアリング
python dscore/score.py \
    -r <reference_rttm_dir> \
    -s <system_rttm_files> \
    --collar 0
```

## プロジェクト構造

### コアモジュール (`diarizen/`)
- `models/eend/`: EEND (End-to-End Neural Diarization) モデルの実装
  - `model_fbank_conformer.py`: フィルターバンク特徴量ベース
  - `model_wavlm_conformer.py`: WavLMベース (frozen/updated両対応)
  - `model_pyannote.py`: Pyannoteベースライン
- `models/module/`: Conformerなどのニューラルネットワークモジュール
- `models/pruning/`: 構造化プルーニングの実装
- `pipelines/inference.py`: Hugging Face統合の推論パイプライン
- `clustering/`: クラスタリングアルゴリズム (AHC, VBx)
- `trainer_*.py`: 学習用トレーナークラス (single-opt, dual-opt, distill-prune)

### レシピ (`recipes/`)
- `diar_ssl/`: 基本的な自己教師あり学習ベースの学習
  - `conf/`: 設定ファイル (fbank, wavlm_frozen, wavlm_updated, pyannote_baseline)
  - `dataset.py`: データローダー実装
  - `run_single_opt.py` / `run_dual_opt.py`: 学習スクリプト
  - `infer_avg.py`: 複数チェックポイントの平均化推論
  - `run_stage.sh`: 学習・推論・評価の統合スクリプト

- `diar_ssl_pruning/`: プルーニング付き学習
  - `apply_pruning.py`: プルーニングの適用
  - `run_distill_prune.py`: 蒸留とプルーニングの同時学習
  - `get_wavlm_from_finetuned.py`: ファインチューニング済みモデルからWavLM抽出

### 外部モジュール
- `pyannote-audio/`: pyannoteライブラリ (サブディレクトリとして統合)
- `dscore/`: ダイアライゼーション評価用スコアリングツール (サブモジュール)

## アーキテクチャの重要ポイント

### Dual Optimizerの仕組み
WavLM-updatedモデルは2つの最適化器を使用:
- **optimizer_small** (lr=2e-5): WavLMの事前学習済みパラメータ用
- **optimizer_big** (lr=1e-3): Conformerなどの新規パラメータ用

これにより、事前学習済み特徴抽出器を適切に更新しながら、新規モジュールを効率的に学習します。

### データフォーマット
- **wav.scp**: 音声ファイルのパスリスト (`<uttid> <filepath>`)
- **rttm**: リファレンスのダイアライゼーション結果
- **uem**: 評価区間の定義 (Unpartitioned Evaluation Map)

### 設定ファイル (.toml)
TOMLファイルで以下を定義:
- モデルアーキテクチャ (`[model]`)
- トレーナー設定 (`[trainer]`)
- データセット設定 (`[train_dataset]`, `[validate_dataset]`)
- 最適化器設定 (`[optimizer_*]`)

設定変更前に必ずパスを環境に合わせて更新すること:
- `wavlm_src`: WavLM事前学習モデルのパス
- データセットのscp/rttm/uemファイルパス
- `embedding_model`: 埋め込みモデルのパス (pyannoteディレクトリに配置)

## 開発時の注意事項

### 学習前の準備
1. データセット準備: wav.scp, rttm, uemファイルをrecipes/diar_ssl/data/に配置
2. WavLM事前学習モデルのダウンロード (WavLM-Base+.pt など)
3. 埋め込みモデルのダウンロード (wespeaker-voxceleb-resnet34-LMなど)
4. run_stage.shとtomlファイル内のパスを実環境に合わせて修正

### チェックポイント管理
- チェックポイントは`exp/<config_name>/`に保存
- `avg_ckpt_num`で複数チェックポイントの平均化が可能
- `val_metric`で検証指標を選択 (Loss or DER)
- `val_mode`で選択方法を指定 (best, prev, center)

### GPUメモリ管理
- バッチサイズとGPU数を環境に合わせて調整
- gradient_accumulation_stepsで実効バッチサイズを増加可能
- chunk_size=8が標準 (8秒チャンク)

### コードスタイル
- Ruffを使用 (line-length=119)
- Google docstring規約
- pre-commitフック設定済み

## 日本語対応と多言語サポート

### 学習データセットの言語構成
DiariZenは以下の言語で学習されています：
- **英語**: AMI、NOTSOFAR-1、VoxConverse
- **中国語**: AISHELL-4、AliMeeting
- **アラビア語**: RAMC
- **多言語混在**: MSDWild、DIHARD3

### 日本語対応について
❌ **公式には日本語データセットで学習されていません**

✅ **ただし理論的には使用可能：**
- WavLMは言語非依存の音響特徴を抽出（136言語をカバーする拡張あり）
- 話者分離は言語の意味理解が不要で、声質・韻律の違いに基づく
- 日本語音声でも動作する可能性が高い

⚠️ **注意点：**
- 日本語での性能評価は未実施
- 実用前に日本語音声でのテストを推奨
- 必要に応じて日本語データでファインチューニング

### 日本語話者分離の代替案
- **AssemblyAI**: 日本語公式サポート（2024年〜）
- **AWS Transcribe**: 日本語対応済み
- **Google Cloud Speech-to-Text**: 日本語対応済み
- **Whisper + Pyannote.audio**: 日本語実装例多数

## 他システムとの性能比較

### ベンチマーク結果（DER: 低いほど良い）

| データセット | Pyannote 3.1 | DiariZen-Base | DiariZen-Large | 改善率 |
|-------------|-------------|--------------|---------------|-------|
| AMI (英語会議) | 22.4% | 15.8% | **14.0%** | **37.5%↓** |
| AISHELL-4 (中国語) | 12.2% | 10.7% | **9.8%** | **19.7%↓** |
| AliMeeting (中国語) | 24.4% | 14.1% | **12.5%** | **48.8%↓** |
| RAMC (アラビア語) | 22.2% | 11.4% | **11.0%** | **50.5%↓** |
| VoxConverse (英語) | 11.3% | 9.7% | **9.2%** | **18.6%↓** |

**平均: Pyannote 3.1比で約30-40%のエラー削減**

### DiariZenの主な優位性

1. **圧倒的な精度向上**
   - 全データセットでPyannote 3.1を大幅に上回る性能
   - AMI、AISHELL-4で2024年9月時点の最高性能達成

2. **データ効率**
   - 学習データ量5%（14.4時間）でPyannote 3.1（全データ）を上回る
   - WavLMの事前学習により少量データでも高精度

3. **汎用性**
   - ドメイン適応なしで複数データセットに対応
   - 同一パラメータで英語・中国語・アラビア語に対応

4. **モデル圧縮**
   - 構造化プルーニングで80%のパラメータ削減
   - 精度を維持しながら計算効率向上

5. **リアルタイム性**
   - GPU使用時はリアルタイムの数倍〜数十倍高速

### 他システムとの位置づけ

| システム | 用途 | 特徴 |
|---------|------|------|
| **DiariZen** | 研究・高精度 | 最高水準の精度、OSS、非商用 |
| **Pyannote 3.1** | 汎用・安定 | 老舗、安定性重視 |
| **AssemblyAI** | 商用・多言語 | 日本語公式対応、APIサービス |
| **NeMo (NVIDIA)** | 高速処理 | GPU最適化、速度重視 |
| **Whisper+Pyannote** | 音声認識統合 | 文字起こし+話者分離 |

## 引用文献

このプロジェクトを使用する場合は、以下の論文を引用してください：

```bibtex
@inproceedings{han2025leveraging,
  title={Leveraging self-supervised learning for speaker diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  booktitle={Proc. ICASSP},
  year={2025}
}
```
