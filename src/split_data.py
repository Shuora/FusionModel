import logging
import os
import random
import socket

import dpkt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'CICAndMal2017')
PCAP_DIR_NAME = 'pcap_data'
IMG_DIR_NAME = 'image_data'
TRAIN_RATIO = 0.8
SEED = 42
PCAP_EXTENSIONS = ('.pcap',)
EXCLUDE_DIRS = {PCAP_DIR_NAME, IMG_DIR_NAME}

def get_real_path(path):
    """Handle case sensitivity on Windows or just verify existence."""
    if os.path.exists(path):
        return path
    # Try capitalizing
    cap_path = path.capitalize()
    if os.path.exists(cap_path):
        return cap_path
    # Try fully lower
    lower_path = path.lower()
    if os.path.exists(lower_path):
        return lower_path
    return path

def ip_to_str(ip_bytes):
    try:
        return socket.inet_ntoa(ip_bytes).replace('.', '-')
    except OSError:
        return None

def extract_sessions(pcap_path):
    sessions = {}
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for _, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.UnpackError, ValueError):
                continue
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue
            if isinstance(ip.data, dpkt.tcp.TCP):
                proto = 'TCP'
                transport = ip.data
            elif isinstance(ip.data, dpkt.udp.UDP):
                proto = 'UDP'
                transport = ip.data
            else:
                continue
            src_ip = ip_to_str(ip.src)
            dst_ip = ip_to_str(ip.dst)
            if not src_ip or not dst_ip:
                continue
            payload = transport.data
            if not payload:
                continue
            key = (proto, src_ip, transport.sport, dst_ip, transport.dport)
            sessions.setdefault(key, bytearray()).extend(payload)
    return sessions

def iter_family_pcaps(dataset_root):
    for top_level in os.listdir(dataset_root):
        top_path = os.path.join(dataset_root, top_level)
        if not os.path.isdir(top_path):
            continue
        for family in os.listdir(top_path):
            family_path = os.path.join(top_path, family)
            if not os.path.isdir(family_path):
                continue
            if family in EXCLUDE_DIRS:
                continue
            files = [f for f in os.listdir(family_path) if f.lower().endswith(PCAP_EXTENSIONS)]
            if files:
                yield family, family_path, files

def split_dataset():
    random.seed(SEED)

    # Resolve paths
    dataset_root = get_real_path(DATASET_ROOT)
    img_root = os.path.join(dataset_root, IMG_DIR_NAME)
    pcap_root = os.path.join(dataset_root, PCAP_DIR_NAME)

    if not os.path.exists(dataset_root):
        logger.error(f"Dataset root '{DATASET_ROOT}' not found.")
        return

    os.makedirs(pcap_root, exist_ok=True)
    os.makedirs(img_root, exist_ok=True)

    families = sorted(iter_family_pcaps(dataset_root), key=lambda x: x[0].lower())
    logger.info(f"Found {len(families)} families to process.")

    for family, src_data_family, files in tqdm(families, desc="Splitting families"):
        random.shuffle(files)

        split_idx = int(len(files) * TRAIN_RATIO)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        splits = {
            'Train': train_files,
            'Test': test_files
        }

        for split_name, split_files in splits.items():
            dst_data_family = os.path.join(pcap_root, split_name, family)
            dst_img_family = os.path.join(img_root, split_name, family)
            os.makedirs(dst_data_family, exist_ok=True)
            os.makedirs(dst_img_family, exist_ok=True)

            for pcap_file in split_files:
                src_pcap_path = os.path.join(src_data_family, pcap_file)
                try:
                    sessions = extract_sessions(src_pcap_path)
                except Exception as e:
                    logger.error(f"Error reading {src_pcap_path}: {e}")
                    continue
                if not sessions:
                    continue
                pcap_base = os.path.splitext(os.path.basename(src_pcap_path))[0]
                for (proto, src_ip, src_port, dst_ip, dst_port), bin_data in sessions.items():
                    if not bin_data:
                        continue
                    session_name = f"{pcap_base}.{proto}_{src_ip}_{src_port}_{dst_ip}_{dst_port}"
                    bin_path = os.path.join(dst_data_family, session_name + '.bin')
                    img_path = os.path.join(dst_img_family, session_name + '.png')
                    try:
                        with open(bin_path, 'wb') as f:
                            f.write(bin_data)
                    except Exception as e:
                        logger.error(f"Error writing {bin_path}: {e}")
                        continue
                    # Image file will be generated by ssl_tls_rgb_image.py

    logger.info("Splitting Completed!")

if __name__ == '__main__':
    split_dataset()
