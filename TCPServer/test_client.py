#!/usr/bin/env python3
"""
Test client for TinyTransformer-Zynq lwIP server.
Sends real model weights (xform_weights.bin) and checks hardware output
against the fixed-point Python reference.

Usage:
  python test_client.py                    # weights + random input smoke test
  python test_client.py <ip> [port]
  python test_client.py --verify <ip>      # verify 8 test vectors against reference
"""

import socket
import struct
import time
import sys
import os
import zlib
import json

SERVER_IP = "192.168.0.102"
SERVER_PORT = 5000
SRC_IP = "192.168.0.150"

PROTOCOL_MAGIC = 0x54524E53
PKT_TYPE_HELLO = 0x01
PKT_TYPE_HELLO_ACK = 0x02
PKT_TYPE_CONFIG = 0x10
PKT_TYPE_CONFIG_ACK = 0x11
PKT_TYPE_WEIGHTS = 0x20
PKT_TYPE_WEIGHTS_ACK = 0x21
PKT_TYPE_INFERENCE = 0x30
PKT_TYPE_RESULT = 0x31
PKT_TYPE_ERROR = 0xFF

NUM_WEIGHTS = 522
NUM_LOGITS = 2
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "xform_weights.bin")
REFS_PATH = os.path.join(os.path.dirname(__file__), "pl", "xform_core_hls", "ref", "test_vectors.json")

def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF

def send_packet(sock, pkt_type, seq, payload=b""):
    header = struct.pack("<IHHI", PROTOCOL_MAGIC, pkt_type, seq, len(payload))
    header += struct.pack("<I", crc32(payload))
    sock.sendall(header + payload)

def recv_packet(sock):
    header_data = b""
    while len(header_data) < 16:
        chunk = sock.recv(16 - len(header_data))
        if not chunk:
            return None
        header_data += chunk
    magic, pkt_type, seq, length, crc_val = struct.unpack("<IHHII", header_data)
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            return None
        payload += chunk
    return magic, pkt_type, seq, length, crc_val, payload

def load_weights():
    path = os.path.abspath(WEIGHTS_PATH)
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) == NUM_WEIGHTS * 2, f"bad weight file size {len(data)}, want {NUM_WEIGHTS*2}"
    return data

def load_reference_outputs():
    with open(REFS_PATH) as f:
        vecs = json.load(f)
    out = []
    for v in vecs:
        out.append((v["in"], v["q12"]))
    return out

def test_connection(verify=False):
    print(f"Connecting to {SERVER_IP}:{SERVER_PORT}...")
    if SRC_IP:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((SRC_IP, 0))
        sock.settimeout(8)
        sock.connect((SERVER_IP, SERVER_PORT))
    else:
        sock = socket.create_connection((SERVER_IP, SERVER_PORT), timeout=8)
    sock.settimeout(8)
    print("Connected!")

    # HELLO
    print("\n--- HELLO ---")
    send_packet(sock, PKT_TYPE_HELLO, 1)
    resp = recv_packet(sock)
    assert resp and resp[1] == PKT_TYPE_HELLO_ACK, f"HELLO failed: {resp}"
    print("HELLO_ACK OK")

    # CONFIG
    print("\n--- CONFIG ---")
    config_payload = struct.pack("<IIIII", 2, 4, 4, 16, 1)
    send_packet(sock, PKT_TYPE_CONFIG, 2, config_payload)
    resp = recv_packet(sock)
    assert resp and resp[1] == PKT_TYPE_CONFIG_ACK, f"CONFIG failed: {resp}"
    print("CONFIG_ACK OK")

    # WEIGHTS (real model)
    print("\n--- WEIGHTS ---")
    weights = load_weights()
    print(f"Loading {len(weights)}B from {os.path.abspath(WEIGHTS_PATH)}")
    send_packet(sock, PKT_TYPE_WEIGHTS, 3, weights)
    resp = recv_packet(sock)
    assert resp and resp[1] == PKT_TYPE_WEIGHTS_ACK, f"WEIGHTS failed: {resp}"
    print("WEIGHTS_ACK OK")

    # INFERENCE
    if verify:
        print("\n--- INFERENCE (verify 8 reference vectors) ---")
        refs = load_reference_outputs()
        pass_count = 0
        for i, (inp, exp_logits) in enumerate(refs):
            inf_payload = struct.pack("<16h", *inp)
            send_packet(sock, PKT_TYPE_INFERENCE, 10 + i, inf_payload)
            resp = recv_packet(sock)
            assert resp, f"no response for vector {i}"
            if resp[1] == PKT_TYPE_ERROR:
                print(f"vec {i}: ERROR 0x{resp[4]:02x}")
                continue
            assert resp[1] == PKT_TYPE_RESULT, f"vec {i}: unexpected type 0x{resp[1]:02x}"
            logits = struct.unpack("<2h", resp[5][:4])
            latency = struct.unpack("<I", resp[5][4:8])[0]
            match = list(logits) == list(exp_logits)
            if match:
                pass_count += 1
            print(f"vec {i}: hw={list(logits)} ref={exp_logits} lat={latency}us {'PASS' if match else 'FAIL'}")
        print(f"\nVerify result: {pass_count}/8 exact match")
    else:
        print("\n--- INFERENCE (smoke test) ---")
        inf_payload = struct.pack("<16h", 1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4, 0, 0, 0, 0)
        send_packet(sock, PKT_TYPE_INFERENCE, 4, inf_payload)
        resp = recv_packet(sock)
        assert resp and resp[1] == PKT_TYPE_RESULT, f"INFERENCE failed: {resp}"
        logits = struct.unpack("<2h", resp[5][:4])
        latency = struct.unpack("<I", resp[5][4:8])[0]
        print(f"RESULT OK: logits={list(logits)}, latency={latency}us")

    sock.close()
    print("\nTest complete!")

if __name__ == "__main__":
    verify = False
    args = [a for a in sys.argv[1:]]
    if "--verify" in args:
        verify = True
        args.remove("--verify")
    if "--src" in args:
        i = args.index("--src")
        SRC_IP = args[i + 1]
        args = args[:i] + args[i + 2:]
    if args:
        SERVER_IP = args[0]
    if len(args) > 1:
        SERVER_PORT = int(args[1])
    test_connection(verify=verify)