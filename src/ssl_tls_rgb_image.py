from __future__ import annotations
import argparse
import inspect
import logging
from pathlib import Path
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **kwargs):
        return iterable
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
BIN_EXTENSIONS = ('.bin',)
IMAGE_SIZE = (28, 28)
R_HEAD_SIZE = 1024
G_HEAD_SIZE = 1024
SESSION_SIZE = 28 * 28

def require_image_dependencies() -> None:
    if np is None:
        raise ModuleNotFoundError('numpy is required to generate RGB images')
    if Image is None:
        raise ModuleNotFoundError('Pillow is required to generate RGB images')

def resolve_roots(dataset_root: Path, input_dir: Path | None=None, output_dir: Path | None=None) -> tuple[Path, Path]:
    dataset_root = Path(dataset_root)
    resolved_input = Path(input_dir) if input_dir else dataset_root / 'pcap_data'
    resolved_output = Path(output_dir) if output_dir else dataset_root / 'image_data'
    return (resolved_input, resolved_output)

def get_bin_files(root_dir: Path) -> list[Path]:
    return sorted((path for path in Path(root_dir).rglob('*') if path.is_file() and path.suffix.lower() in BIN_EXTENSIONS))

def extract_gray_channel(bin_data: bytes):
    require_image_dependencies()
    arr = np.frombuffer(bin_data, dtype=np.uint8)
    if len(arr) < SESSION_SIZE:
        arr = np.pad(arr, (0, SESSION_SIZE - len(arr)), 'constant')
    else:
        arr = arr[:SESSION_SIZE]
    return arr.reshape(IMAGE_SIZE)

def extract_r_channel(bin_data: bytes):
    require_image_dependencies()
    head = bin_data[:R_HEAD_SIZE]
    r = np.frombuffer(head, dtype=np.uint8)
    if len(r) < SESSION_SIZE:
        r = np.pad(r, (0, SESSION_SIZE - len(r)), 'constant')
    else:
        r = r[:SESSION_SIZE]
    return r

def extract_g_channel(bin_data: bytes):
    require_image_dependencies()
    handshake = bin_data[:G_HEAD_SIZE]
    handshake_arr = np.frombuffer(handshake, dtype=np.uint8)
    cipher_suite_diversity = int(np.unique(handshake_arr[:32]).size) if handshake_arr.size >= 32 else 0
    sni_bytes = handshake_arr[32:96] if handshake_arr.size >= 96 else np.array([], dtype=np.uint8)
    sni_entropy = 0.0
    if sni_bytes.size and sni_bytes.any():
        probs = np.bincount(sni_bytes, minlength=256) / float(sni_bytes.size)
        probs = probs[probs > 0]
        sni_entropy = float(-np.sum(probs * np.log2(probs)))
    cert_anomaly = int(handshake_arr.max() - handshake_arr.min()) if handshake_arr.size else 0
    g = np.zeros(SESSION_SIZE, dtype=np.uint8)
    g[0] = cipher_suite_diversity % 256
    g[1] = int(sni_entropy * 32) % 256
    g[2] = cert_anomaly % 256
    remaining_space = SESSION_SIZE - 3
    if handshake_arr.size > remaining_space:
        g[3:] = handshake_arr[:remaining_space]
    else:
        g[3:3 + handshake_arr.size] = handshake_arr
    return g

def extract_b_channel(bin_data: bytes):
    require_image_dependencies()
    arr = np.frombuffer(bin_data, dtype=np.uint8)
    pkt_size = 1500
    pkts = [arr[i:i + pkt_size] for i in range(0, len(arr), pkt_size)]
    pkt_lens = [len(p) for p in pkts]
    mean_len = int(np.mean(pkt_lens)) if pkt_lens else 0
    interval_var = 0
    if len(pkts) > 1:
        intervals = [int(pkts[i][0]) - int(pkts[i - 1][0]) for i in range(1, len(pkts))]
        interval_var = int(np.var(intervals)) % 256
    duration = len(pkts) % 256
    b = np.zeros(SESSION_SIZE, dtype=np.uint8)
    b[0] = mean_len % 256
    b[1] = interval_var
    b[2] = duration
    fill_len = min(len(pkt_lens), SESSION_SIZE - 3)
    if fill_len:
        values = np.clip(np.array(pkt_lens[:fill_len], dtype=np.int32), 0, 255).astype(np.uint8)
        positions = np.linspace(3, SESSION_SIZE - 1, num=fill_len, dtype=int)
        b[positions] = values
    return b

def get_output_path(bin_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel_path = Path(bin_path).relative_to(Path(input_dir))
    return Path(output_dir) / rel_path.with_suffix('.png')

def process_bin_file(bin_path: Path, input_dir: Path, output_dir: Path, mode: str='rgb') -> Path:
    require_image_dependencies()
    with Path(bin_path).open('rb') as f:
        bin_data = f.read()
    if mode == 'gray':
        gray = extract_gray_channel(bin_data)
        img = Image.fromarray(gray.astype(np.uint8), mode='L')
    else:
        r = extract_r_channel(bin_data)
        g = extract_g_channel(bin_data)
        b = extract_b_channel(bin_data)
        rgb = np.stack([r, g, b], axis=1).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        img = Image.fromarray(rgb.astype(np.uint8))
    out_path = get_output_path(Path(bin_path), Path(input_dir), Path(output_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    img.save(tmp_path, format='PNG')
    tmp_path.replace(out_path)
    return out_path

def setup_logging(log_file: Path) -> Path:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(str(log_file), encoding='utf-8'), logging.StreamHandler()]
    kwargs = {'level': logging.INFO, 'format': '%(asctime)s - %(levelname)s - %(message)s'}
    if 'force' in inspect.signature(logging.basicConfig).parameters:
        kwargs['force'] = True
    logging.basicConfig(handlers=handlers, **kwargs)
    return log_file

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert processed session bins into RGB or Gray images.')
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--mode', choices=['rgb', 'gray'], default='rgb', help='Image mode: rgb (3-channel) or gray (1-channel)')
    parser.add_argument('--log_file', default='')
    return parser

def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    (input_dir, output_dir) = resolve_roots(dataset_root, Path(args.input_dir) if args.input_dir else None, Path(args.output_dir) if args.output_dir else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else dataset_root / 'metadata' / 'ssl_tls_rgb_image.log'
    resolved_log = setup_logging(log_file)
    logger.info('Image preprocessing log file: %s', resolved_log)
    logger.info('Mode: %s', args.mode)
    bin_files = get_bin_files(input_dir)
    logger.info('Found %s bin files.', len(bin_files))
    skipped = 0
    processed = 0
    progress = tqdm(bin_files, desc=f'Processing images ({args.mode})', ncols=100, unit='file')
    for bin_path in progress:
        out_path = get_output_path(bin_path, input_dir, output_dir)
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            if hasattr(progress, 'set_postfix'):
                progress.set_postfix(processed=processed, skipped=skipped, refresh=False)
            continue
        process_bin_file(bin_path, input_dir, output_dir, mode=args.mode)
        processed += 1
        if hasattr(progress, 'set_postfix'):
            progress.set_postfix(processed=processed, skipped=skipped, refresh=False)
    logger.info('Done. processed=%s, skipped=%s, total=%s', processed, skipped, len(bin_files))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())