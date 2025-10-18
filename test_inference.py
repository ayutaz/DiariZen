"""
DiariZenモデルの動作確認スクリプト
"""
import os
import sys
from diarizen.pipelines.inference import DiariZenPipeline

# UTF-8出力を設定
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# モデルのキャッシュディレクトリを指定
cache_dir = os.path.join(os.getcwd(), 'models')

print("=" * 60)
print("DiariZen Model Test")
print("=" * 60)

# モデルの読み込み
print("\n[1/3] Loading model...")
print("Model: BUT-FIT/diarizen-wavlm-large-s80-md")
diar_pipeline = DiariZenPipeline.from_pretrained(
    "BUT-FIT/diarizen-wavlm-large-s80-md",
    cache_dir=cache_dir
)
print("OK: Model loaded successfully")

# 音声ファイルのパス
audio_file = './example/EN2002a_30s.wav'
print(f"\n[2/3] Processing audio: {audio_file}")

# ダイアライゼーションの実行
diar_results = diar_pipeline(audio_file)
print("OK: Diarization completed")

# 結果の表示
print(f"\n[3/3] Results:")
print("-" * 60)
print(f"{'Start':>10} | {'End':>10} | Speaker")
print("-" * 60)
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"{turn.start:>9.1f}s | {turn.end:>9.1f}s | speaker_{speaker}")
print("-" * 60)

print("\nOK: Test completed successfully!")
print("=" * 60)
