"""TCG-only experiment for Homework 4.
Compares: Raw features baseline vs TCG graph features vs Raw+TCG combined.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboard.compat.proto import event_pb2, summary_pb2

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
if sys.platform == "win32":
    for attr in ("stdout", "stderr"):
        stream = getattr(sys, attr)
        try:
            buf = stream.buffer
        except AttributeError:
            pass
        else:
            setattr(sys, attr, io.TextIOWrapper(buf, encoding="utf-8", errors="replace"))

ROOT = Path(os.environ.get("TUGRAPH3_ROOT", Path(__file__).resolve().parents[1]))
DATASET = ROOT / "Dataset-Unicauca-Version2-87Atts.csv" / "Dataset-Unicauca-Version2-87Atts.csv"
IMPORT_DIR = ROOT / "tugraph_import"
OUTPUT_DIR = ROOT / "outputs" / "tcg_only"
RUNS_DIR = ROOT / "runs" / "tcg_only"
REPORT_DIR = ROOT / "reports"

CRC32C_POLY = 0x82F63B78


def _crc32c_table() -> list[int]:
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ CRC32C_POLY if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return table


CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


class SimpleTensorBoardWriter:
    """Small TFRecord event writer."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        host = os.environ.get("COMPUTERNAME", "windows")
        filename = f"events.out.tfevents.{int(time.time())}.{host}"
        self.path = self.log_dir / filename
        self.file = self.path.open("wb")
        self._write_event(event_pb2.Event(wall_time=time.time(), file_version="brain.Event:2"))

    def _write_record(self, data: bytes) -> None:
        length = struct.pack("<Q", len(data))
        self.file.write(length)
        self.file.write(struct.pack("<I", masked_crc32c(length)))
        self.file.write(data)
        self.file.write(struct.pack("<I", masked_crc32c(data)))

    def _write_event(self, event: event_pb2.Event) -> None:
        self._write_record(event.SerializeToString())

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))])
        self._write_event(event_pb2.Event(wall_time=time.time(), step=int(step), summary=summary))

    def close(self) -> None:
        self.file.flush()
        self.file.close()


@dataclass
class ExperimentConfig:
    scan_rows: int
    top_classes: int
    samples_per_class: int
    embedding_dim: int
    test_size: float
    random_state: int
    epochs: int
    batch_size: int
    causal_window_seconds: int


def endpoint(ip: object, port: object) -> str:
    try:
        port_text = str(int(float(port)))
    except Exception:
        port_text = str(port)
    return f"{ip}:{port_text}"


def safe_numeric_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    excluded = {
        "Flow.ID", "Source.IP", "Destination.IP", "Timestamp", "Label",
        "ProtocolName", "L7Protocol", target_col,
    }
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    x = df[numeric_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True).fillna(0)
    return x.fillna(medians)


def load_balanced_sample(cfg: ExperimentConfig) -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    print(f"Loading first {cfg.scan_rows:,} rows from dataset ...")
    df = pd.read_csv(DATASET, nrows=cfg.scan_rows, low_memory=False, encoding="utf-8")
    df = df.dropna(subset=["Source.IP", "Destination.IP", "Source.Port", "Destination.Port", "ProtocolName"])
    top = df["ProtocolName"].value_counts().head(cfg.top_classes).index.tolist()
    print(f"Top {cfg.top_classes} classes: {top}")
    df = df[df["ProtocolName"].isin(top)].copy()
    sampled = (
        df.groupby("ProtocolName", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), cfg.samples_per_class), random_state=cfg.random_state))
        .sample(frac=1.0, random_state=cfg.random_state)
        .reset_index(drop=True)
    )
    sampled.insert(0, "row_id", [f"flow_{i:07d}" for i in range(len(sampled))])
    sampled["src_endpoint"] = [endpoint(ip, port) for ip, port in zip(sampled["Source.IP"], sampled["Source.Port"])]
    sampled["dst_endpoint"] = [endpoint(ip, port) for ip, port in zip(sampled["Destination.IP"], sampled["Destination.Port"])]
    sampled["timestamp_dt"] = pd.to_datetime(sampled["Timestamp"], format="%d/%m/%Y%H:%M:%S", errors="coerce")
    print("Class distribution:")
    print(sampled["ProtocolName"].value_counts().to_string())
    return sampled


def export_tcg(df: pd.DataFrame, causal_window_seconds: int) -> dict[str, int]:
    """Export TCG with four edge types: CR, PR, DHR, SHR (Liu Zhen paper)."""
    df = df.copy()
    df["_src_port"] = pd.to_numeric(df["Source.Port"], errors="coerce").fillna(-1).astype(int)
    df["_dst_port"] = pd.to_numeric(df["Destination.Port"], errors="coerce").fillna(-1).astype(int)
    df["_protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(int)

    vertices = pd.DataFrame({
        "flow_id": df["row_id"],
        "src_endpoint": df["src_endpoint"],
        "dst_endpoint": df["dst_endpoint"],
        "protocol_name": df["ProtocolName"],
        "timestamp": df["Timestamp"],
    })

    rows = df[["row_id", "Source.IP", "Destination.IP", "_src_port", "_dst_port", "_protocol", "timestamp_dt"]].copy()
    rows.columns = ["row_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp_dt"]
    recs = rows.to_dict("records")

    cr_edges = _build_cr_edges(recs, causal_window_seconds)
    pr_edges = _build_pr_edges(recs, causal_window_seconds)
    dhr_edges = _build_dhr_edges(recs, causal_window_seconds)
    shr_edges = _build_shr_edges(recs, causal_window_seconds)

    counts = {"CR": len(cr_edges), "PR": len(pr_edges), "DHR": len(dhr_edges), "SHR": len(shr_edges)}
    total = sum(counts.values())
    print(f"TCG edges — CR:{counts['CR']} PR:{counts['PR']} DHR:{counts['DHR']} SHR:{counts['SHR']} total:{total}")

    IMPORT_DIR.mkdir(exist_ok=True)
    vertices.to_csv(IMPORT_DIR / "tcg_vertices_flow.csv", index=False, encoding="utf-8")
    _save_edges(cr_edges, "tcg_edges_CR.csv", ("src_flow", "dst_flow", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "delta_seconds"))
    _save_edges(pr_edges, "tcg_edges_PR.csv", ("src_flow", "dst_flow", "shared_ip", "delta_seconds"))
    _save_edges(dhr_edges, "tcg_edges_DHR.csv", ("src_flow", "dst_flow", "shared_ip", "src_port_f1", "src_port_f2", "delta_seconds"))
    _save_edges(shr_edges, "tcg_edges_SHR.csv", ("src_flow", "dst_flow", "shared_ip", "shared_port", "delta_seconds"))
    return counts


def _save_edges(edges: list[dict[str, object]], filename: str, columns: tuple[str, ...]) -> None:
    df_out = pd.DataFrame(edges, columns=list(columns)) if edges else pd.DataFrame(columns=list(columns))
    df_out.to_csv(IMPORT_DIR / filename, index=False, encoding="utf-8")


def _build_cr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    from collections import defaultdict
    index: dict[tuple, list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        key = (r["protocol"], str(r["src_ip"]), str(r["dst_ip"]), int(r["src_port"]), int(r["dst_port"]))
        rev_key = (r["protocol"], str(r["dst_ip"]), str(r["src_ip"]), int(r["dst_port"]), int(r["src_port"]))
        for other in index.get(rev_key, []):
            delta = abs((ts - other["timestamp_dt"]).total_seconds())
            if delta <= window_sec and r["row_id"] != other["row_id"]:
                edges.append({
                    "src_flow": other["row_id"], "dst_flow": r["row_id"],
                    "src_ip": str(r["src_ip"]), "src_port": int(r["src_port"]),
                    "dst_ip": str(r["dst_ip"]), "dst_port": int(r["dst_port"]),
                    "protocol": r["protocol"], "delta_seconds": float(delta),
                })
                break
        index[key].append(r)
    return edges


def _build_pr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    from collections import defaultdict
    index: dict[str, list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        for prev in index.get(str(r["src_ip"]), []):
            delta = (ts - prev["timestamp_dt"]).total_seconds()
            if 0 <= delta <= window_sec and r["row_id"] != prev["row_id"]:
                edges.append({
                    "src_flow": prev["row_id"], "dst_flow": r["row_id"],
                    "shared_ip": str(r["src_ip"]), "delta_seconds": float(delta),
                })
                break
        index[str(r["dst_ip"])].append(r)
    return edges


def _build_dhr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    from collections import defaultdict
    index: dict[str, list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        src_ip = str(r["src_ip"])
        src_port = int(r["src_port"])
        for prev in index[src_ip]:
            prev_port = int(prev["src_port"])
            if prev_port != src_port:
                delta = (ts - prev["timestamp_dt"]).total_seconds()
                if 0 <= delta <= window_sec and r["row_id"] != prev["row_id"]:
                    edges.append({
                        "src_flow": prev["row_id"], "dst_flow": r["row_id"],
                        "shared_ip": src_ip, "src_port_f1": prev_port,
                        "src_port_f2": src_port, "delta_seconds": float(delta),
                    })
                    break
        index[src_ip].append(r)
    return edges


def _build_shr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    from collections import defaultdict
    index: dict[tuple[str, int], list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        key = (str(r["src_ip"]), int(r["src_port"]))
        for prev in index[key]:
            delta = (ts - prev["timestamp_dt"]).total_seconds()
            if 0 <= delta <= window_sec and r["row_id"] != prev["row_id"]:
                edges.append({
                    "src_flow": prev["row_id"], "dst_flow": r["row_id"],
                    "shared_ip": key[0], "shared_port": key[1],
                    "delta_seconds": float(delta),
                })
                break
        index[key].append(r)
    return edges


def build_nx_graph(left: Iterable[str], right: Iterable[str], directed: bool = False):
    import networkx as nx
    g = nx.DiGraph() if directed else nx.Graph()
    for u, v in zip(left, right):
        if u != v:
            g.add_edge(u, v)
    return g


def _alias_setup(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(probs)
    prob = probs * n
    alias = np.zeros(n, dtype=np.int32)
    small, large = [], []
    for i, p in enumerate(prob):
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        s = small.pop()
        l = large.pop()
        alias[s] = l
        prob[l] = prob[l] + prob[s] - 1.0
        if prob[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
    while large:
        prob[large.pop()] = 1.0
    while small:
        prob[small.pop()] = 1.0
    return prob.astype(np.float32), alias


def _alias_draw(prob: np.ndarray, alias: np.ndarray, rng: np.random.RandomState) -> int:
    n = len(prob)
    col = rng.randint(0, n)
    if rng.rand() < prob[col]:
        return col
    return alias[col]


def node2vec_random_walks(g, walk_length: int, num_walks: int, p: float, q: float, seed: int = 42) -> list[list[str]]:
    import networkx as nx
    rng = np.random.RandomState(seed)
    nodes = list(g.nodes())
    walks: list[list[str]] = []

    alias_nodes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for node in nodes:
        neighbors = list(g.neighbors(node))
        if not neighbors:
            alias_nodes[node] = (np.array([0.0]), np.array([0], dtype=np.int32))
            continue
        probs = np.ones(len(neighbors), dtype=np.float64) / len(neighbors)
        alias_nodes[node] = _alias_setup(probs)

    alias_edges: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _get_alias_edge(t: str, v: str) -> tuple[np.ndarray, np.ndarray]:
        key = (t, v)
        if key in alias_edges:
            return alias_edges[key]
        neighbors = list(g.neighbors(v))
        if not neighbors:
            probs = np.array([1.0])
            j_arr = np.array([0], dtype=np.int32)
            alias_edges[key] = (probs, j_arr)
            return alias_edges[key]
        probs = np.zeros(len(neighbors), dtype=np.float64)
        for i, x in enumerate(neighbors):
            if x == t:
                probs[i] = 1.0 / p
            elif g.has_edge(x, t) or (isinstance(g, nx.Graph) and g.has_edge(t, x)):
                probs[i] = 1.0
            else:
                probs[i] = 1.0 / q
        probs /= probs.sum()
        alias_edges[key] = _alias_setup(probs)
        return alias_edges[key]

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            if g.degree(start) == 0:
                continue
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_neighbors = list(g.neighbors(cur))
                if not cur_neighbors:
                    break
                if len(walk) == 1:
                    probs, j_arr = alias_nodes[cur]
                else:
                    probs, j_arr = _get_alias_edge(walk[-2], cur)
                idx = _alias_draw(probs, j_arr, rng)
                if idx < len(cur_neighbors):
                    walk.append(cur_neighbors[idx])
                else:
                    break
            if len(walk) >= 2:
                walks.append(walk)
    return walks


def node2vec_embeddings(
    left: Iterable[str], right: Iterable[str], dim: int,
    walk_length: int = 7, num_walks: int = 10,
    p: float = 0.3, q: float = 0.7, directed: bool = False,
    epochs: int = 5,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    g = build_nx_graph(left, right, directed=directed)
    nodes = sorted(g.nodes())
    if len(nodes) < 2:
        return {n: np.zeros(dim, dtype=np.float32) for n in nodes}, np.zeros((len(nodes), dim), dtype=np.float32)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    walks = node2vec_random_walks(g, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
    if len(walks) < 10:
        adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for u, v in zip(left, right):
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                adj[i, j] = 1.0
                if not directed:
                    adj[j, i] = 1.0
        n_components = max(1, min(dim, min(adj.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        emb = svd.fit_transform(adj).astype(np.float32)
        if n_components < dim:
            emb = np.pad(emb, ((0, 0), (0, dim - n_components)))
        return {node: emb[i] for node, i in node_to_idx.items()}, emb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_walks = [[node_to_idx[n] for n in w] for w in walks]

    window = 5
    pairs: list[tuple[int, int]] = []
    for walk in idx_walks:
        for i, center in enumerate(walk):
            start = max(0, i - window)
            end = min(len(walk), i + window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, walk[j]))

    if len(pairs) == 0:
        emb = np.random.randn(len(nodes), dim).astype(np.float32) * 0.01
        return {node: emb[i] for node, i in node_to_idx.items()}, emb

    n_vocab = len(nodes)
    emb_in = nn.Embedding(n_vocab, dim, sparse=False)
    emb_out = nn.Embedding(n_vocab, dim, sparse=False)
    emb_in.weight.data.normal_(0, 0.01)
    emb_out.weight.data.normal_(0, 0.01)

    node_freq = np.zeros(n_vocab, dtype=np.float64)
    for w in idx_walks:
        for ni in w:
            node_freq[ni] += 1
    node_freq = node_freq ** 0.75
    node_freq /= node_freq.sum()

    neg_samples = 5
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.01)

    pair_arr = np.array(pairs, dtype=np.int64)
    batch_size = 2048
    n_batches = max(1, len(pair_arr) // batch_size)

    for epoch in range(epochs):
        np.random.shuffle(pair_arr)
        total_loss = 0.0
        for b in range(n_batches):
            batch = pair_arr[b * batch_size : (b + 1) * batch_size]
            centers = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            contexts = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            neg = torch.multinomial(
                torch.tensor(node_freq, dtype=torch.float32, device=device),
                len(centers) * neg_samples, replacement=True,
            ).view(len(centers), neg_samples)

            emb_c = emb_in(centers)
            emb_pos = emb_out(contexts)
            emb_neg = emb_out(neg)

            pos_score = (emb_c * emb_pos).sum(dim=1).sigmoid().log()
            neg_score = (-(emb_c.unsqueeze(1) * emb_neg).sum(dim=2)).sigmoid().log().sum(dim=1)
            loss = -(pos_score + neg_score).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

    emb = emb_in.weight.detach().cpu().numpy().astype(np.float32)
    return {node: emb[i] for node, i in node_to_idx.items()}, emb


def build_tcg_features(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build TCG flow embeddings from 4 edge types. Returns (features_df, edge_type_embeddings)."""
    edge_files = {
        "CR": ("tcg_edges_CR.csv", ("src_flow", "dst_flow")),
        "PR": ("tcg_edges_PR.csv", ("src_flow", "dst_flow")),
        "DHR": ("tcg_edges_DHR.csv", ("src_flow", "dst_flow")),
        "SHR": ("tcg_edges_SHR.csv", ("src_flow", "dst_flow")),
    }

    tcg_dim = max(4, cfg.embedding_dim // 4)  # smaller dim per edge type, 4 types total
    all_embeddings: dict[str, dict[str, np.ndarray]] = {}
    tcg_arrays = []

    for label, (filename, (src_col, dst_col)) in edge_files.items():
        edge_path = IMPORT_DIR / filename
        print(f"  TCG/{label}: ", end="", flush=True)
        if edge_path.exists():
            edge_df = pd.read_csv(edge_path, encoding="utf-8")
            print(f"{len(edge_df)} edges", end="", flush=True)
            if len(edge_df) > 5:
                emb, _ = node2vec_embeddings(
                    edge_df[src_col].astype(str), edge_df[dst_col].astype(str),
                    dim=tcg_dim, walk_length=7, num_walks=10, p=0.3, q=0.7, directed=True,
                )
            else:
                emb = {fid: np.zeros(tcg_dim, dtype=np.float32) for fid in df["row_id"]}
        else:
            emb = {fid: np.zeros(tcg_dim, dtype=np.float32) for fid in df["row_id"]}
            print("file not found", end="", flush=True)

        all_embeddings[label] = emb
        flow_emb = np.vstack([emb.get(fid, np.zeros(tcg_dim, dtype=np.float32)) for fid in df["row_id"]])
        tcg_arrays.append(flow_emb)
        print(f" → {flow_emb.shape[1]}d embeddings")

    tcg_combined = np.hstack(tcg_arrays)
    cols = []
    for ci, label in enumerate(["CR", "PR", "DHR", "SHR"]):
        for j in range(tcg_dim):
            cols.append(f"tcg_{label}_{j}")
    tcg_df = pd.DataFrame(tcg_combined, columns=cols)
    return tcg_df, all_embeddings


def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    degree_counts = pd.concat([df["src_endpoint"], df["dst_endpoint"]]).value_counts()
    structural = pd.DataFrame({
        "src_degree": df["src_endpoint"].map(degree_counts).fillna(0).to_numpy(),
        "dst_degree": df["dst_endpoint"].map(degree_counts).fillna(0).to_numpy(),
    })
    return structural


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        hidden = min(256, max(64, in_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
    class_names: list[str], cfg: ExperimentConfig, writer: SimpleTensorBoardWriter, tag_prefix: str,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    model = SimpleMLP(x_train.shape[1], len(class_names)).to(device)

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-5)

    best_f1 = -math.inf
    best_state = None
    patience = max(5, cfg.epochs // 4)
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)

        scheduler.step()
        metrics, _ = eval_torch(model, x_test, y_test, device)
        train_loss = total_loss / len(train_ds)
        writer.add_scalar(f"{tag_prefix}/loss/train", train_loss, epoch)
        writer.add_scalar(f"{tag_prefix}/metrics/test_accuracy", metrics["accuracy"], epoch)
        writer.add_scalar(f"{tag_prefix}/metrics/test_f1", metrics["f1_weighted"], epoch)

        print(f"  [{tag_prefix}] epoch {epoch:02d}/{cfg.epochs} loss={train_loss:.4f} acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f}")

        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{tag_prefix}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, y_pred = eval_torch(model, x_test, y_test, device)
    return metrics


def eval_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return metric_dict(y, pred), pred


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
    }


def write_artifacts(name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / f"{name}_classification_report.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_comparison(results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Plot comparison bar chart for Raw vs TCG vs Raw+TCG."""
    feature_groups = list(results.keys())
    model_names = list(results[feature_groups[0]].keys())

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        f1_values = [results[fg][model_name]["f1_weighted"] for fg in feature_groups]
        acc_values = [results[fg][model_name]["accuracy"] for fg in feature_groups]
        x = np.arange(len(feature_groups))
        width = 0.35
        ax.bar(x - width / 2, acc_values, width, label="Accuracy")
        ax.bar(x + width / 2, f1_values, width, label="Weighted F1")
        ax.set_xticks(x)
        ax.set_xticklabels(feature_groups, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(model_name)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tcg_feature_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TCG-only experiment for network traffic classification.")
    parser.add_argument("--scan-rows", type=int, default=200_000)
    parser.add_argument("--top-classes", type=int, default=6)
    parser.add_argument("--samples-per-class", type=int, default=3000)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--causal-window-seconds", type=int, default=60)
    args = parser.parse_args()
    cfg = ExperimentConfig(**vars(args))

    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    for d in [IMPORT_DIR, OUTPUT_DIR, RUNS_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    writer = SimpleTensorBoardWriter(RUNS_DIR)
    (RUNS_DIR / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("TCG Experiment — Homework 4")
    print(f"Config: {asdict(cfg)}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading and sampling data ...")
    df = load_balanced_sample(cfg)
    n_samples = len(df)
    writer.add_scalar("data/n_samples", n_samples, 0)

    # 2. Build TCG graph
    print("\n[2/5] Building TCG graph (CR, PR, DHR, SHR edges) ...")
    tcg_counts = export_tcg(df, cfg.causal_window_seconds)
    for edge_type, count in tcg_counts.items():
        writer.add_scalar(f"tcg/edge_count_{edge_type}", count, 0)

    # 3. Raw features baseline
    print("\n[3/5] Preparing feature sets ...")
    raw_features = safe_numeric_frame(df, target_col="ProtocolName")
    print(f"  Raw features: {raw_features.shape[1]} dimensions")

    tcg_features, tcg_embeddings = build_tcg_features(df, cfg)
    print(f"  TCG features: {tcg_features.shape[1]} dimensions")

    structural = build_structural_features(df)
    print(f"  Structural features: {structural.shape[1]} dimensions")

    # 4. Prepare train/test splits
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ProtocolName"])
    class_names = label_encoder.classes_.tolist()
    print(f"\n  Classes: {class_names}")

    # Build 3 feature configurations
    feature_sets: dict[str, pd.DataFrame] = {
        "Raw": raw_features.reset_index(drop=True),
        "TCG": pd.concat([tcg_features.reset_index(drop=True), structural.reset_index(drop=True)], axis=1),
        "Raw+TCG": pd.concat([raw_features.reset_index(drop=True), tcg_features.reset_index(drop=True), structural.reset_index(drop=True)], axis=1),
    }

    # 5. Train and evaluate
    print("\n[4/5] Training classifiers for each feature set ...")
    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for fg_name, features_df in feature_sets.items():
        print(f"\n--- Feature Group: {fg_name} ({features_df.shape[1]} dims) ---")
        features_clean = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        x = scaler.fit_transform(features_clean).astype(np.float32)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )

        fg_results: dict[str, dict[str, float]] = {}

        # Decision Tree
        print("  Training DecisionTree ...")
        dt = DecisionTreeClassifier(max_depth=18, min_samples_leaf=3, random_state=42)
        dt.fit(x_train, y_train)
        dt_pred = dt.predict(x_test)
        dt_metrics = metric_dict(y_test, dt_pred)
        fg_results["DecisionTree"] = dt_metrics
        write_artifacts(f"tcg_{fg_name}_DecisionTree", y_test, dt_pred, class_names)
        print(f"    DT: acc={dt_metrics['accuracy']:.4f} f1={dt_metrics['f1_weighted']:.4f}")

        # KNN
        print("  Training KNN ...")
        knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        knn.fit(x_train, y_train)
        knn_pred = knn.predict(x_test)
        knn_metrics = metric_dict(y_test, knn_pred)
        fg_results["KNN"] = knn_metrics
        write_artifacts(f"tcg_{fg_name}_KNN", y_test, knn_pred, class_names)
        print(f"    KNN: acc={knn_metrics['accuracy']:.4f} f1={knn_metrics['f1_weighted']:.4f}")

        # Random Forest
        print("  Training RandomForest ...")
        rf = RandomForestClassifier(n_estimators=160, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=42)
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        rf_metrics = metric_dict(y_test, rf_pred)
        fg_results["RandomForest"] = rf_metrics
        write_artifacts(f"tcg_{fg_name}_RandomForest", y_test, rf_pred, class_names)
        print(f"    RF: acc={rf_metrics['accuracy']:.4f} f1={rf_metrics['f1_weighted']:.4f}")

        # MLP with TensorBoard monitoring
        print(f"  Training MLP ({cfg.epochs} epochs) ...")
        mlp_metrics = train_mlp(x_train, x_test, y_train, y_test, class_names, cfg, writer, f"mlp/{fg_name}")
        fg_results["MLP"] = mlp_metrics
        print(f"    MLP: acc={mlp_metrics['accuracy']:.4f} f1={mlp_metrics['f1_weighted']:.4f}")

        all_results[fg_name] = fg_results

        # Log summary metrics
        for model_name, m in fg_results.items():
            writer.add_scalar(f"summary/{fg_name}/{model_name}/accuracy", m["accuracy"], 0)
            writer.add_scalar(f"summary/{fg_name}/{model_name}/f1_weighted", m["f1_weighted"], 0)

    writer.close()

    # 6. Generate report
    print("\n[5/5] Generating report and figures ...")
    plot_comparison(all_results)

    # Calculate TCG gain
    print("\n" + "=" * 60)
    print("TCG Feature Contribution Analysis")
    print("=" * 60)
    for model_name in ["DecisionTree", "KNN", "RandomForest", "MLP"]:
        raw_f1 = all_results["Raw"][model_name]["f1_weighted"]
        tcg_f1 = all_results["TCG"][model_name]["f1_weighted"]
        combined_f1 = all_results["Raw+TCG"][model_name]["f1_weighted"]
        tcg_gain = combined_f1 - raw_f1
        direction = "▲ +" if tcg_gain > 0 else "▼ "
        print(f"  {model_name:15s}: Raw={raw_f1:.4f}  TCG-only={tcg_f1:.4f}  Raw+TCG={combined_f1:.4f}  Gain={direction}{tcg_gain:.4f}")

    # Save all results
    summary = {}
    for fg_name, fg_results in all_results.items():
        for model_name, metrics in fg_results.items():
            summary[f"{fg_name}_{model_name}"] = metrics

    (OUTPUT_DIR / "tcg_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).T.to_csv(OUTPUT_DIR / "tcg_results.csv", encoding="utf-8-sig")

    # Save detailed comparison data
    comparison_data = {
        "config": asdict(cfg),
        "tcg_edge_counts": tcg_counts,
        "results": {
            fg: {model: metrics for model, metrics in models.items()}
            for fg, models in all_results.items()
        },
    }
    (OUTPUT_DIR / "tcg_full_results.json").write_text(json.dumps(comparison_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"TensorBoard: tensorboard --logdir {RUNS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
