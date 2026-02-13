import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 新增进度条
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing_CICAndMal2017.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'CICAndMal2017')
INPUT_DIR = os.path.join(DATASET_ROOT, 'pcap_data')
OUTPUT_DIR = os.path.join(DATASET_ROOT, 'image_data')
BIN_EXTENSIONS = ('.bin',)
IMAGE_SIZE = (28, 28)  # 输出图像尺寸
R_HEAD_SIZE = 512        # R通道：原始字节流头部优先 512B
G_HEAD_SIZE = 1024       # G通道：握手明文数据优先 1KB
SESSION_SIZE = 28 * 28   # 每个通道像素数（统一 28x28）

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_bin_files(root_dir):
    bin_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(BIN_EXTENSIONS):
                bin_files.append(os.path.join(root, f))
    return bin_files

def extract_r_channel(bin_data):
    # R通道：优先载荷头 512B，保留底层空间分布，尾部零填充统一尺寸
    head = bin_data[:R_HEAD_SIZE]
    r = np.frombuffer(head, dtype=np.uint8)
    if len(r) < SESSION_SIZE:
        r = np.pad(r, (0, SESSION_SIZE - len(r)), 'constant')
    else:
        r = r[:SESSION_SIZE]
    return r


def extract_g_channel(bin_data):
    # G通道：握手明文语义 + 语义矩阵编码，优先前 1KB 握手数据
    handshake = bin_data[:G_HEAD_SIZE]
    handshake_padded = handshake.ljust(G_HEAD_SIZE, b'\x00') if len(handshake) < G_HEAD_SIZE else handshake[:G_HEAD_SIZE]
    handshake_arr = np.frombuffer(handshake_padded, dtype=np.uint8)
    # 密码套件组合（简单用前32字节的唯一值个数）
    cipher_suite_diversity = int(np.unique(handshake_arr[:32]).size)
    # SNI域名熵（用前64字节的熵近似）
    sni_bytes = handshake_arr[32:96]
    sni_entropy = 0.0
    if sni_bytes.size and sni_bytes.any():
        probs = np.bincount(sni_bytes, minlength=256) / 64.0
        probs = probs[probs > 0]
        sni_entropy = float(-np.sum(probs * np.log2(probs)))
    # 证书链异常（用明文区最大字节值-最小字节值近似）
    cert_anomaly = int(handshake_arr.max() - handshake_arr.min()) if handshake_arr.size else 0
    # 语义矩阵编码：前三个位置放语义特征，其余按握手明文字节填充
    g = np.zeros(SESSION_SIZE, dtype=np.uint8)
    g[0] = cipher_suite_diversity % 256
    g[1] = int(sni_entropy * 32) % 256
    g[2] = cert_anomaly % 256
    fill_len = min(handshake_arr.size, SESSION_SIZE - 3)
    if fill_len:
        g[3:3 + fill_len] = handshake_arr[:fill_len]
    return g


def extract_b_channel(bin_data):
    # B通道：会话行为统计特征，优先行为指标后分散填充长度序列
    arr = np.frombuffer(bin_data, dtype=np.uint8)
    pkt_size = 1500
    pkts = [arr[i:i + pkt_size] for i in range(0, len(arr), pkt_size)]
    pkt_lens = [len(p) for p in pkts]
    mean_len = int(np.mean(pkt_lens)) if pkt_lens else 0
    if len(pkts) > 1:
        intervals = [int(pkts[i][0]) - int(pkts[i - 1][0]) for i in range(1, len(pkts))]
        interval_var = int(np.var(intervals))
    else:
        interval_var = 0
    duration = len(pkts)
    b = np.zeros(SESSION_SIZE, dtype=np.uint8)
    b[0] = mean_len % 256
    b[1] = interval_var % 256
    b[2] = duration % 256
    fill_len = min(len(pkt_lens), SESSION_SIZE - 3)
    if fill_len:
        values = np.clip(np.array(pkt_lens[:fill_len], dtype=np.int32), 0, 255).astype(np.uint8)
        # 将长度分散到全图，避免只集中在左上角
        positions = np.linspace(3, SESSION_SIZE - 1, num=fill_len, dtype=int)
        b[positions] = values
    return b

def get_output_path(bin_path, output_dir):
    rel_path = os.path.relpath(bin_path, INPUT_DIR)
    rel_no_ext = os.path.splitext(rel_path)[0]
    return os.path.join(output_dir, rel_no_ext + '.png')

def process_bin_file(bin_path, output_dir):
    with open(bin_path, 'rb') as f:
        bin_data = f.read()
    r = extract_r_channel(bin_data)
    g = extract_g_channel(bin_data)
    b = extract_b_channel(bin_data)
    rgb = np.stack([r, g, b], axis=1).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    img = Image.fromarray(rgb.astype(np.uint8))
    out_path = get_output_path(bin_path, output_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + '.tmp.png'
    img.save(tmp_path, format='PNG')
    os.replace(tmp_path, out_path)
    logger.info(f"Saved: {out_path}")

def main():
    bin_files = get_bin_files(INPUT_DIR)
    logger.info(f"Found {len(bin_files)} bin files.")
    skipped = 0
    processed = 0
    for bin_path in tqdm(bin_files, desc="Processing", ncols=80):
        out_path = get_output_path(bin_path, OUTPUT_DIR)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue
        process_bin_file(bin_path, OUTPUT_DIR)
        processed += 1
    logger.info(f"Done. processed={processed}, skipped={skipped}, total={len(bin_files)}")

if __name__ == '__main__':
    main()
