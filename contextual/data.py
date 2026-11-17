"""Real reply ancestry, strict validation, and leakage-safe grouped splits."""
import hashlib
import json
import shutil
import zipfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

from .config import location, project_path, write_json


def fingerprint(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class MissingParentError(ValueError):
    """A source conversation references an utterance absent from the corpus."""


class EmptyMessageError(ValueError):
    """A source conversation contains a message without usable text."""


class InvalidAncestryError(ValueError):
    """A source conversation contains a cycle or a cross-conversation link."""


def download(cfg):
    """Download atomically, preserve provenance, extract only data (no scripts)."""
    dc = cfg["data"]
    spec = dc["cga"]
    root = location(cfg, "raw")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec["archive"]
    def valid_zip(path):
        try:
            with zipfile.ZipFile(path) as zf:
                return zf.testzip() is None and any(Path(n).name == "utterances.jsonl" for n in zf.namelist())
        except (OSError, zipfile.BadZipFile):
            return False

    if archive.exists() and not valid_zip(archive):
        archive.replace(archive.with_suffix(".corrupt.zip"))
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        attempts = dc.get("download_attempts", 3)
        for attempt in range(attempts):
            try:
                print(f"Downloading corpus (attempt {attempt + 1}/{attempts})", flush=True)
                with requests.get(spec["url"], stream=True, timeout=dc["download_timeout_seconds"]) as response:
                    response.raise_for_status()
                    size = 0
                    with partial.open("wb") as out:
                        for chunk in response.iter_content(dc["download_chunk_bytes"]):
                            size += len(chunk)
                            if size > dc["max_archive_bytes"]:
                                raise ValueError("Dataset exceeds configured download size limit")
                            out.write(chunk)
                            print(f"Downloaded {size:,} bytes", end="\r", flush=True)
                if not valid_zip(partial):
                    raise ValueError("Dataset is incomplete or is not a valid corpus ZIP")
                partial.replace(archive)
                break
            except (requests.RequestException, OSError, ValueError) as exc:
                if attempt + 1 == attempts:
                    raise RuntimeError(f"Download failed after {attempts} attempts. Rerun this download cell. "
                                       f"Existing prepared data is preserved. Source: {spec['url']}. Error: {exc}") from exc
                time.sleep(dc.get("download_retry_seconds", 2))
    else:
        print(f"Using verified local archive: {archive}", flush=True)
    destination = root / spec["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        if sum(info.file_size for info in zf.infolist()) > dc["max_archive_bytes"]:
            raise ValueError("Expanded archive exceeds configured size limit")
        for info in zf.infolist():
            name = Path(info.filename).name
            if name not in {"utterances.jsonl", "conversations.json", "index.json", "corpus.json"}:
                continue
            target = destination / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            with zf.open(info) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            temporary.replace(target)
    if not (destination / "utterances.jsonl").exists():
        raise ValueError("CGA archive has no utterances.jsonl")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(dc["download_chunk_bytes"]), b""):
            digest.update(block)
    write_json(destination / "provenance.json", {"url": spec["url"], "sha256": digest.hexdigest()})
    return destination


def ancestor_chain(nodes, target_id, max_turns):
    """Only ancestors and target, never siblings or future descendants."""
    if target_id not in nodes:
        raise ValueError(f"Unknown target id: {target_id}")
    chain, seen = [], set()
    current = target_id
    conversation = nodes[target_id]["conversation_id"]
    while current is not None:
        if current in seen:
            raise InvalidAncestryError(f"Reply cycle at {current}")
        if current not in nodes:
            raise MissingParentError(f"Missing parent {current} of {target_id}")
        seen.add(current)
        node = nodes[current]
        if node["conversation_id"] != conversation:
            raise InvalidAncestryError("Parent crosses conversations")
        if not isinstance(node["text"], str) or not node["text"].strip():
            raise EmptyMessageError(f"Empty message: {current}")
        chain.append(node)
        current = node.get("parent_id")
    return list(reversed(chain[:max_turns]))


def examples_from_nodes(records, cfg, source, *, skip_missing_parents=False, stats=None):
    nodes = {str(r["id"]): {**r, "id": str(r["id"]),
             "parent_id": str(r["parent_id"]) if r.get("parent_id") is not None else None,
             "conversation_id": str(r["conversation_id"])} for r in records}
    if len(nodes) != len(records):
        raise ValueError(f"Duplicate message IDs in {source}")
    examples = []
    for node in nodes.values():
        label = node.get("label")
        if label is None:
            continue
        if label not in (0, 1):
            raise ValueError(f"Expected binary target label in {source}")
        try:
            chain = ancestor_chain(nodes, node["id"], cfg["model"]["max_turns"])
        except MissingParentError:
            if not skip_missing_parents:
                raise
            if stats is not None:
                stats["skipped_missing_ancestry"] = stats.get("skipped_missing_ancestry", 0) + 1
            continue
        except EmptyMessageError:
            if not skip_missing_parents:
                raise
            if stats is not None:
                stats["skipped_empty_ancestry"] = stats.get("skipped_empty_ancestry", 0) + 1
            continue
        except InvalidAncestryError:
            if not skip_missing_parents:
                raise
            if stats is not None:
                stats["skipped_invalid_ancestry"] = stats.get("skipped_invalid_ancestry", 0) + 1
            continue
        examples.append({"id": f"{source}:{node['id']}", "group": f"{source}:{node['conversation_id']}",
                         "source": source, "texts": [n["text"] for n in chain],
                         "message_ids": [n["id"] for n in chain], "label": int(label)})
    return examples


def load_cga(cfg, stats=None):
    root = location(cfg, "raw") / cfg["data"]["cga"]["directory"]
    path = root / "utterances.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python run_pipeline.py --download first.")
    # Discard large syntactic parses immediately instead of retaining the entire corpus.
    raw = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta", {})
            row["meta"] = {key: meta.get(key) for key in
                           ("is_section_header", cfg["data"]["cga"]["label_field"])}
            raw.append({key: row.get(key) for key in
                        ("id", "text", "meta", "conversation_id", "root", "reply-to", "reply_to")})
    headers = {str(r["id"]) for r in raw if r.get("meta", {}).get("is_section_header")}
    nodes = []
    for r in raw:
        if str(r["id"]) in headers:
            continue
        parent = r.get("reply-to") if r.get("reply-to") is not None else r.get("reply_to")
        parent = None if parent is None or str(parent) in headers else str(parent)
        nodes.append({"id": str(r["id"]), "parent_id": parent, "text": r["text"],
                      "conversation_id": r.get("conversation_id") or r.get("root"),
                      "label": r.get("meta", {}).get(cfg["data"]["cga"]["label_field"])})
    # The released corpus omits some referenced parents. Do not turn their
    # descendants into artificial roots or invent context for them.
    examples = examples_from_nodes(nodes, cfg, "cga_wiki", skip_missing_parents=True, stats=stats)
    # Group by article as well as by conversation, so paired threads stay together.
    conversations_path = root / "conversations.json"
    if conversations_path.exists():
        conversations = json.loads(conversations_path.read_text(encoding="utf-8"))
        for ex in examples:
            cid = ex["group"].split(":", 1)[1]
            meta = conversations.get(cid, {}).get("meta", {})
            if meta.get("page_id") is not None:
                ex["group"] = f"cga_wiki:page:{meta['page_id']}"
    return examples


def grouped_split(examples, cfg):
    """Union conversations sharing any message text before splitting.

    This prevents both duplicate target leakage and a held-out target appearing
    as a training ancestor. Duplicate messages are connected, not relabelled.
    """
    from sklearn.model_selection import GroupShuffleSplit
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            next_node = parent[x]
            parent[x] = root
            x = next_node
        return root

    owner = {}
    for ex in examples:
        g = ex["group"]
        for text in ex["texts"]:
            key = fingerprint(text)
            if key in owner:
                parent[find(g)] = find(owner[key])
            owner[key] = g
    groups = [find(ex["group"]) for ex in examples]
    if len(set(groups)) < 3:
        raise ValueError("At least three independent conversation groups are required")
    ids = np.arange(len(examples))
    d = cfg["data"]
    seed = cfg["training"]["seed"]
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=d["test_fraction"], random_state=seed).split(ids, groups=groups))
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=d["validation_fraction"] / (1-d["test_fraction"]), random_state=seed).split(train_val, groups=np.array(groups)[train_val]))
    for split, indices in (("train", train_val[train]), ("val", train_val[val]), ("test", test)):
        for i in indices:
            examples[i]["split"] = split
        if {examples[i]["label"] for i in indices} != {0, 1}:
            raise ValueError(f"{split} does not contain both classes; supply more independent labelled data")
    return examples


def prepare(cfg):
    if cfg['data'].get('prepared_only'):
        raise ValueError('This is a cleaned dataset version. Run python -m contextual.preprocessing as described in RUN_AND_REPORT.md.')
    examples = []
    cga_stats = {}
    path = location(cfg, "raw") / cfg["data"]["cga"]["directory"] / "utterances.jsonl"
    if path.exists():
        examples.extend(load_cga(cfg, stats=cga_stats))
    custom = project_path(cfg["data"]["custom_threads"])
    if custom.exists() and cfg["data"].get("include_custom_threads", True):
        examples.extend(examples_from_nodes(read_jsonl(custom), cfg, "custom_threads"))
    real_context_count = sum(len(e["texts"]) > 1 for e in examples)
    if cfg["data"]["require_real_context"] and not real_context_count:
        raise ValueError("No real parent-child training examples. Run --download or supply custom_threads.jsonl. "
                         "Isolated comments cannot train context understanding.")
    import pandas as pd
    sources = []
    for spec in cfg["data"]["local_sources"]:
        path = project_path(spec["path"])
        if not path.exists():
            if spec["optional"]:
                sources.append({"source": spec["source"], "status": "missing_optional", "path": str(path)})
                continue
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        for i, row in frame.iterrows():
            text, label = row["text"], row[spec["label"]]
            if not isinstance(text, str) or not text.strip() or label not in (0, 1):
                continue
            conv_id = row.get("conv_id")
            group = str(conv_id) if pd.notna(conv_id) and str(conv_id).strip() else fingerprint(text)
            examples.append({"id": f"{spec['source']}:{i}", "group": f"{spec['source']}:{group}",
                             "source": spec["source"], "texts": [text], "message_ids": [str(i)], "label": int(label)})
        sources.append({"source": spec["source"], "status": "loaded", "path": str(path)})
    if not examples:
        raise ValueError("No labelled data available")
    examples = grouped_split(examples, cfg)
    destination = location(cfg, "prepared")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for ex in examples:
            stream.write(json.dumps(ex, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    manifest = {"examples": len(examples), "real_context_examples": real_context_count,
                "cga_skipped_missing_ancestry": cga_stats.get("skipped_missing_ancestry", 0),
                "cga_skipped_empty_ancestry": cga_stats.get("skipped_empty_ancestry", 0),
                "cga_skipped_invalid_ancestry": cga_stats.get("skipped_invalid_ancestry", 0),
                "splits": dict(Counter(e["split"] for e in examples)), "sources": sources,
                "source_counts": dict(Counter(e["source"] for e in examples)),
                "split_policy": "New grouped splits; all shared message text and CGA articles stay in one split. Not the original benchmark splits.",
                "task": "Binary target toxicity; CGA personal attacks and local toxicity labels are related but not identical constructs."}
    write_json(location(cfg, "manifest"), manifest)
    return manifest
