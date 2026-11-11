"""Build a real DisarmRAG edited Contriever for the B10 experiment.

The upstream archive contains the MEND editor and its NQ edit data, but its
``train_retriever_editor.sh`` points at a module that is not shipped and its
training code assumes a cached ``facebook/contriever`` model.  This wrapper
uses the upstream ``EditTrainer`` and MEND implementation, while adapting the
model path, output directory, and attack instruction for a server-local run.

The script never overwrites an existing output directory unless ``--force`` is
provided.  The resulting directory is a normal HuggingFace Contriever
directory (``config.json`` plus model weights) and can be passed to the
DisarmRAG loader used by ``integrated_service.py``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_source(source_zip: Path, work_dir: Path) -> tuple[Path, str]:
    """Extract one content-addressed copy and return its root and hash."""
    source_hash = sha256_file(source_zip)
    extract_dir = work_dir / f"source-{source_hash[:12]}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        roots = {name.split("/", 1)[0] for name in names if name}
        if len(roots) != 1:
            raise ValueError(f"expected one source root in {source_zip}, found {sorted(roots)}")
        root_name = next(iter(roots))
        source_root = extract_dir / root_name
        marker = extract_dir / ".ready"
        if not marker.exists() or not source_root.is_dir():
            for name in names:
                target = (extract_dir / name).resolve()
                if extract_dir.resolve() not in target.parents and target != extract_dir.resolve():
                    raise ValueError(f"unsafe archive member: {name}")
            archive.extractall(extract_dir)
            marker.write_text(root_name + "\n", encoding="utf-8")
    return source_root, source_hash


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected a non-empty JSON list: {path}")
    return rows


def set_attack_instruction(rows: list[dict[str, Any]], instruction: str) -> None:
    for row in rows:
        rewrite = row.get("rewrite_query")
        if not isinstance(rewrite, dict) or not rewrite.get("query"):
            raise ValueError("each edit row needs rewrite_query.query")
        rewrite["target_instruction"] = instruction


def patch_optional_upstream_imports(source_root: Path) -> None:
    """Keep the Contriever editor independent of optional vision packages.

    The archive's ``editing.trainer.__init__`` eagerly imports BLIP-2 and
    multimodal trainers.  Those modules require ``timm`` and other image
    dependencies even though MEND/Contriever never uses them.  The dataset
    initializer has the same issue for image/video datasets.  Replace only
    the extracted copy's initializers with the modules required for this run;
    the original ZIP remains untouched.
    """
    editing_init = source_root / "editing/__init__.py"
    trainer_init = source_root / "editing/trainer/__init__.py"
    dataset_init = source_root / "editing/dataset/__init__.py"
    if not editing_init.is_file() or not trainer_init.is_file() or not dataset_init.is_file():
        raise FileNotFoundError("DisarmRAG editing package initializers are missing")
    trainer_minimal = (
        "from .training_hparams import *\n"
        "from .algs import *\n"
        "from .EditTrainer import *\n"
        "from .BaseTrainer import *\n"
    )
    trainer_current = trainer_init.read_text(encoding="utf-8")
    if ".blip2_models" in trainer_current or ".MultimodalTrainer" in trainer_current:
        trainer_init.write_text(trainer_minimal, encoding="utf-8")

    # Only InstructDataset is needed for Contriever editing.  The upstream
    # dataset initializer also imports image/video datasets (and their
    # optional torchvision/av dependencies), so keep those out of this run.
    dataset_current = dataset_init.read_text(encoding="utf-8")
    if "from .instruct import InstructDataset" not in dataset_current or ".coco_caption" in dataset_current:
        dataset_init.write_text("from .instruct import InstructDataset\n", encoding="utf-8")

    # The top-level initializer otherwise asks the reduced dataset package for
    # ZsreDataset.  Keep only the symbols used by this Contriever/MEND run.
    editing_current = editing_init.read_text(encoding="utf-8")
    if "from .dataset import ZsreDataset" in editing_current or ".edit_contriever_mend" in editing_current:
        editing_init.write_text(
            "from .dataset import InstructDataset\n"
            "from .trainer import EditTrainer, MENDTrainingHparams\n",
            encoding="utf-8",
        )


def patch_instruct_dataset(source_root: Path) -> None:
    """Make the Contriever dataset independent of optional ST imports.

    The upstream InstructDataset imports SentenceTransformer at module import
    time, although the Contriever branch only needs AutoTokenizer.  This is
    incompatible with a legacy Transformers stack and is unnecessary here.
    Keep the import local to the one branch that actually uses it.
    """
    path = source_root / "editing/dataset/instruct.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("from sentence_transformers import SentenceTransformer\n", "")
    old = """            if config.model_name == "sentence-transformers/msmarco-roberta-base-ance-firstp":
                model = SentenceTransformer(config.model_name)
"""
    new = """            if config.model_name == "sentence-transformers/msmarco-roberta-base-ance-firstp":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(config.model_name)
"""
    if old in text:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def patch_local_model_loader(source_root: Path, clean_model: Path) -> None:
    """Patch only the imported trainer module through a runtime monkeypatch.

    The upstream BaseTrainer imports ``get_model`` into its module namespace.
    Replacing that symbol lets the official MEND code load a server-local
    Contriever directory while retaining the original model/editor classes.
    """
    base_module = importlib.import_module("editing.trainer.BaseTrainer")
    contriever_module = importlib.import_module("src.contriever_src.contriever")
    Contriever = contriever_module.Contriever

    def get_local_model(config, pre_edited_model=None):
        model = Contriever.from_pretrained(str(clean_model), trust_remote_code=True)
        if pre_edited_model is not None:
            model.load_state_dict(pre_edited_model.state_dict())
        dropout = getattr(config, "dropout", None)
        if dropout is not None:
            for module in model.modules():
                if hasattr(module, "dropout") and isinstance(module.dropout, float):
                    module.dropout = dropout
        return model

    base_module.get_model = get_local_model


def build_hparams(source_root: Path, clean_model: Path, work_dir: Path,
                  device: str, max_iters: int, val_steps: int) -> Path:
    import yaml

    template = source_root / "hparams/TRAINING/MEND/contriever.yaml"
    if not template.is_file():
        raise FileNotFoundError(template)
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config.update({
        # Keep this identifier because the upstream trainer branches on it;
        # the runtime get_model adapter loads the local directory below.
        "model_name": "facebook/contriever",
        "tokenizer_name": str(clean_model),
        "device": device,
        "results_dir": str(work_dir / "mend-results"),
        "max_iters": int(max_iters),
        "max_epochs": None,
        "val_steps": int(val_steps),
        "batch_size": 1,
        "val_batch_size": 1,
        "archive": None,
        "save": True,
        "debug": False,
        "eval_only": False,
        "final_eval": True,
    })
    output = work_dir / "hparams_b10_contriever.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def write_prompt_artifact(source_root: Path, instruction: str) -> Path:
    path = source_root / "results/coop_results/evolution_scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_dict = {f"AP{i}": part for i, part in enumerate(instruction.split(), 1)}
    path.write_text(json.dumps([
        {"round": 0, "top_p2": [{"prompt2": prompt_dict}]}
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the upstream DisarmRAG MEND editor and export an edited Contriever"
    )
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--clean-model", required=True, type=Path,
                        help="local facebook/contriever directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory for the edited HuggingFace model")
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="persistent audit/training directory")
    parser.add_argument("--train-limit", type=int, default=0,
                        help="0 uses all upstream edit rows")
    parser.add_argument("--edit-limit", type=int, default=1,
                        help="number of target queries to edit sequentially")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--val-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-instruction", default="",
                        help="override the instruction embedded in the upstream NQ edit data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="allow writing into an existing output directory")
    args = parser.parse_args()

    if not args.source_zip.is_file():
        raise FileNotFoundError(args.source_zip)
    if not args.clean_model.is_dir():
        raise FileNotFoundError(args.clean_model)
    if args.edit_limit < 1 or args.max_iters < 1 or args.val_steps < 1:
        raise ValueError("edit-limit, max-iters and val-steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"output directory is non-empty: {args.output_dir}; use a new path or --force"
        )

    random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or args.output_dir.parent / f".{args.output_dir.name}-work"
    work_dir.mkdir(parents=True, exist_ok=True)

    source_root, source_hash = extract_source(args.source_zip, work_dir)
    patch_optional_upstream_imports(source_root)
    patch_instruct_dataset(source_root)
    train_path = source_root / f"data/instruct_{args.dataset}_train.json"
    eval_path = source_root / f"data/instruct_{args.dataset}_eval.json"
    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(f"missing upstream edit data for {args.dataset}")

    train_rows = load_rows(train_path)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if not train_rows:
        raise ValueError("training data is empty after --train-limit")
    instruction = args.target_instruction.strip() or str(
        train_rows[0]["rewrite_query"].get("target_instruction", "")
    ).strip()
    if not instruction:
        raise ValueError("target instruction is empty; pass --target-instruction")
    set_attack_instruction(train_rows, instruction)
    eval_rows = load_rows(eval_path)
    set_attack_instruction(eval_rows, instruction)
    write_prompt_artifact(source_root, instruction)

    # Import the upstream package only after its extracted root is first on
    # sys.path; this prevents a different installed `editing` package from
    # being selected.
    sys.path.insert(0, str(source_root))
    import torch
    from transformers import AutoTokenizer
    from editing.dataset import InstructDataset
    from editing.trainer import EditTrainer, MENDTrainingHparams

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is false")
    patch_local_model_loader(source_root, args.clean_model)
    hparams_path = build_hparams(
        source_root, args.clean_model, work_dir, args.device,
        args.max_iters, min(args.val_steps, len(eval_rows)),
    )
    config = MENDTrainingHparams.from_hparams(str(hparams_path))
    train_ds = InstructDataset(str(train_path), config=config, loaded_data=train_rows)
    val_ds = InstructDataset(str(eval_path), config=config, loaded_data=eval_rows)

    save_name = "b10_mend_editor"
    trainer = EditTrainer(config=config, train_set=train_ds, val_set=val_ds,
                          save_name=save_name)
    print(json.dumps({
        "stage": "mend_training",
        "train_rows": len(train_rows),
        "max_iters": args.max_iters,
        "save_path": str(trainer.save_path_using),
        "source_root": str(source_root),
    }, ensure_ascii=False), flush=True)
    trainer.run()
    mend_archive = Path(trainer.save_path_using)
    if not mend_archive.is_file():
        raise RuntimeError(f"MEND training did not produce an archive: {mend_archive}")

    # Apply the official MEND editor to one or more target questions.  Each
    # next edit starts from the previous edited Contriever, so the final model
    # contains a reproducible cumulative edit set.
    edited_model = None
    edited_queries: list[str] = []
    for index, source_row in enumerate(train_rows[:args.edit_limit], 1):
        row = copy.deepcopy(source_row)
        query = str(row["rewrite_query"]["query"])
        row["rewrite_query"]["target_instruction"] = instruction
        edit_config = MENDTrainingHparams.from_hparams(str(hparams_path))
        edit_config.archive = str(mend_archive)
        edit_config.eval_only = True
        edit_config.max_iters = 0
        edit_config.val_steps = 1
        edit_config.final_eval = False
        edit_ds = InstructDataset(str(eval_path), config=edit_config,
                                  loaded_data=[row])
        edit_trainer = EditTrainer(
            config=edit_config,
            train_set=train_ds,
            val_set=edit_ds,
            save_name=f"b10_edit_{index:04d}",
            pre_edited_model=edited_model,
        )
        # BaseTrainer loads the trained MEND archive during construction.
        # The upstream evaluate_edit() expects a legacy per-edit archive at
        # save_path_using, which is not created for this sequential export.
        # Validate directly so the loaded archive edits the current target.
        _, edited_model = edit_trainer.validate(
            steps=edit_config.val_steps, return_edited_model=True
        )
        edited_queries.append(query)
        print(json.dumps({
            "stage": "apply_edit",
            "index": index,
            "query": query,
            "total_edits": len(edited_queries),
        }, ensure_ascii=False), flush=True)

    if edited_model is None:
        raise RuntimeError("no edited model was produced")
    edited_model.to("cpu")
    edited_model.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained(str(args.clean_model)).save_pretrained(args.output_dir)

    metadata = {
        "dataset": args.dataset,
        "source_zip": str(args.source_zip),
        "source_sha256": source_hash,
        "source_root": str(source_root),
        "clean_model": str(args.clean_model),
        "output_dir": str(args.output_dir),
        "target_instruction": instruction,
        "train_rows": len(train_rows),
        "edit_limit": args.edit_limit,
        "edited_queries": edited_queries,
        "max_iters": args.max_iters,
        "device": args.device,
        "mend_archive": str(mend_archive),
        "hparams": str(hparams_path),
        "method": "upstream DisarmRAG MEND editor via EditTrainer",
        "optional_import_patch": "editing/trainer/dataset initializers and InstructDataset restricted to MEND/Contriever modules",
    }
    (args.output_dir / "b10_edit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "complete",
        "output_dir": str(args.output_dir),
        "files": sorted(p.name for p in args.output_dir.iterdir()),
        "metadata": str(args.output_dir / "b10_edit_metadata.json"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
