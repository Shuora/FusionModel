#!/usr/bin/env python3
"""Analyze TLS/SSL proportion correctly - check all TLS record types."""

import struct
from pathlib import Path

# TLS record content types
TLS_CONTENT_TYPES = {
    0x14: 'ChangeCipherSpec',
    0x15: 'Alert',
    0x16: 'Handshake',
    0x17: 'ApplicationData',
    0x18: 'Heartbeat',
}

# Common TLS/SSL ports
TLS_PORTS = {443, 8443, 465, 587, 636, 989, 990, 992, 993, 994, 995, 5061}

def is_tls_record(payload: bytes) -> bool:
    """Check if payload starts with a valid TLS record."""
    if len(payload) < 5:
        return False
    
    content_type = payload[0]
    version_major = payload[1]
    version_minor = payload[2]
    
    # TLS 1.x: content_type (1 byte) + version (2 bytes: 0x03, 0x00-0x04) + length (2 bytes)
    if content_type in TLS_CONTENT_TYPES and version_major == 0x03 and version_minor <= 0x04:
        return True
    
    # SSLv2: length (2 bytes) + content_type (1 byte: 0x80 for handshake) + version (2 bytes)
    if payload[0] == 0x80 and payload[3] == 0x02:  # SSLv2 ClientHello
        return True
    
    return False

def analyze_pcap(pcap_path: Path) -> dict:
    """Analyze a pcap file for TLS/SSL traffic."""
    stats = {
        'total_sessions': 0,
        'tls_sessions': 0,
        'total_packets': 0,
        'tls_packets': 0,
        'tls_port_packets': 0,
        'error': False,
    }
    
    sessions = {}  # key -> has_tls
    pkt_count = 0
    
    try:
        with open(pcap_path, 'rb') as f:
            magic = f.read(4)
            f.seek(0)
            
            if magic in [b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\x4d\x3c\xb2\xa1', b'\xa1\xb2\x3c\x4d']:
                if magic == b'\xd4\xc3\xb2\xa1' or magic == b'\x4d\x3c\xb2\xa1':
                    endian = '<'
                else:
                    endian = '>'
                
                global_header = f.read(24)
                if len(global_header) < 24:
                    stats['error'] = True
                    return stats
                
                while True:
                    header = f.read(16)
                    if not header or len(header) < 16:
                        break
                    ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', header)
                    packet = f.read(incl_len)
                    if len(packet) < incl_len:
                        break
                    
                    pkt_count += 1
                    stats['total_packets'] += 1
                    
                    try:
                        eth_type = struct.unpack('>H', packet[12:14])[0]
                        if eth_type != 0x0800:  # IPv4 only
                            continue
                        
                        ip_start = 14
                        ip_header = packet[ip_start:ip_start+20]
                        if len(ip_header) < 20:
                            continue
                        
                        ip_proto = ip_header[9]
                        ip_src = '.'.join(str(b) for b in ip_header[12:16])
                        ip_dst = '.'.join(str(b) for b in ip_header[16:20])
                        ihl = (ip_header[0] & 0x0F) * 4
                        trans_start = ip_start + ihl
                        
                        if ip_proto == 6:  # TCP
                            trans_header = packet[trans_start:trans_start+20]
                            if len(trans_header) < 20:
                                continue
                            sport = struct.unpack('>H', trans_header[0:2])[0]
                            dport = struct.unpack('>H', trans_header[2:4])[0]
                            data_offset = ((trans_header[12] >> 4) * 4)
                            payload_start = trans_start + data_offset
                            payload = packet[payload_start:]
                        elif ip_proto == 17:  # UDP
                            trans_header = packet[trans_start:trans_start+8]
                            if len(trans_header) < 8:
                                continue
                            sport = struct.unpack('>H', trans_header[0:2])[0]
                            dport = struct.unpack('>H', trans_header[2:4])[0]
                            payload_start = trans_start + 8
                            payload = packet[payload_start:]
                        else:
                            continue
                        
                        if not payload:
                            continue
                        
                        # Normalize session key (sorted endpoints)
                        if (ip_src, sport) < (ip_dst, dport):
                            key = (ip_proto, ip_src, sport, ip_dst, dport)
                        else:
                            key = (ip_proto, ip_dst, dport, ip_src, sport)
                        
                        if key not in sessions:
                            sessions[key] = False
                        
                        # Check TLS by port
                        if sport in TLS_PORTS or dport in TLS_PORTS:
                            stats['tls_port_packets'] += 1
                        
                        # Check TLS by record type
                        if is_tls_record(payload):
                            sessions[key] = True
                            stats['tls_packets'] += 1
                            
                    except Exception:
                        continue
            
            else:
                # Try pcapng
                try:
                    import dpkt
                    with open(pcap_path, 'rb') as fh:
                        reader = dpkt.pcapng.Reader(fh)
                        for ts, buf in reader:
                            pkt_count += 1
                            stats['total_packets'] += 1
                            
                            try:
                                eth = dpkt.ethernet.Ethernet(buf)
                                ip = eth.data
                                if not isinstance(ip, dpkt.ip.IP):
                                    continue
                                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                                    transport = ip.data
                                    payload = transport.data
                                    if not payload:
                                        continue
                                    
                                    if (ip.src, transport.sport) < (ip.dst, transport.dport):
                                        key = (ip.proto, ip.src, transport.sport, ip.dst, transport.dport)
                                    else:
                                        key = (ip.proto, ip.dst, transport.dport, ip.src, transport.sport)
                                    
                                    if key not in sessions:
                                        sessions[key] = False
                                    
                                    if transport.sport in TLS_PORTS or transport.dport in TLS_PORTS:
                                        stats['tls_port_packets'] += 1
                                    
                                    if is_tls_record(payload):
                                        sessions[key] = True
                                        stats['tls_packets'] += 1
                            except Exception:
                                continue
                except Exception:
                    stats['error'] = True
                    return stats
                
    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        stats['error'] = True
        return stats
    
    stats['total_sessions'] = len(sessions)
    stats['tls_sessions'] = sum(1 for v in sessions.values() if v)
    
    return stats

def main():
    source_root = Path('SourceData')
    
    datasets = {
        'ISCX-VPN-NonVPN-2016': source_root / 'ISCX-VPN-NonVPN-2016',
        'USTC-TFC2016': source_root / 'USTC-TFC2016',
        'MTA': source_root / 'MTA',
        'MFCP': source_root / 'MFCP',
    }
    
    grand_total = {
        'total_sessions': 0,
        'tls_sessions': 0,
        'total_packets': 0,
        'tls_packets': 0,
        'tls_port_packets': 0,
        'total_files': 0,
        'error_files': 0,
    }
    
    for dataset_name, dataset_root in datasets.items():
        if not dataset_root.exists():
            print(f"Dataset not found: {dataset_root}")
            continue
            
        pcap_files = sorted(list(dataset_root.rglob('*.pcap')) + list(dataset_root.rglob('*.pcapng')))
        grand_total['total_files'] += len(pcap_files)
        
        dataset_stats = {
            'total_sessions': 0,
            'tls_sessions': 0,
            'total_packets': 0,
            'tls_packets': 0,
            'tls_port_packets': 0,
        }
        
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"PCAP files: {len(pcap_files)}")
        
        for i, pcap_file in enumerate(pcap_files):
            if (i + 1) % 20 == 0 or i == len(pcap_files) - 1:
                print(f"  Processing file {i+1}/{len(pcap_files)}...")
            
            stats = analyze_pcap(pcap_file)
            if stats['error']:
                grand_total['error_files'] += 1
                continue
                
            for k in dataset_stats:
                dataset_stats[k] += stats[k]
        
        for k in grand_total:
            if k not in ['total_files', 'error_files']:
                grand_total[k] += dataset_stats[k]
        
        session_ratio = (dataset_stats['tls_sessions'] / dataset_stats['total_sessions'] * 100 
                        if dataset_stats['total_sessions'] > 0 else 0)
        pkt_ratio = (dataset_stats['tls_packets'] / dataset_stats['total_packets'] * 100 
                    if dataset_stats['total_packets'] > 0 else 0)
        port_ratio = (dataset_stats['tls_port_packets'] / dataset_stats['total_packets'] * 100 
                     if dataset_stats['total_packets'] > 0 else 0)
        
        print(f"Total packets: {dataset_stats['total_packets']}")
        print(f"Total sessions: {dataset_stats['total_sessions']}")
        print(f"TLS sessions (by record type): {dataset_stats['tls_sessions']} ({session_ratio:.1f}%)")
        print(f"TLS packets (by record type): {dataset_stats['tls_packets']} ({pkt_ratio:.1f}%)")
        print(f"TLS packets (by port): {dataset_stats['tls_port_packets']} ({port_ratio:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"GRAND TOTALS")
    print(f"Total PCAP files: {grand_total['total_files']}")
    print(f"Files with errors: {grand_total['error_files']}")
    print(f"Total packets: {grand_total['total_packets']}")
    print(f"Total sessions: {grand_total['total_sessions']}")
    
    if grand_total['total_sessions'] > 0:
        print(f"TLS sessions (by record type): {grand_total['tls_sessions']} ({grand_total['tls_sessions']/grand_total['total_sessions']*100:.1f}%)")
    if grand_total['total_packets'] > 0:
        print(f"TLS packets (by record type): {grand_total['tls_packets']} ({grand_total['tls_packets']/grand_total['total_packets']*100:.1f}%)")
        print(f"TLS packets (by port): {grand_total['tls_port_packets']} ({grand_total['tls_port_packets']/grand_total['total_packets']*100:.1f}%)")

if __name__ == '__main__':
    main()
