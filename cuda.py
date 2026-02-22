import torch
import sys
import io

# 強制將標準輸出編碼設為 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"顯卡名稱: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("目前只能使用 CPU。若要使用 GPU，請檢查驅動或重新安裝對應版本的 PyTorch。")