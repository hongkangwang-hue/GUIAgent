"""Qwen2.5-VL-3B 的 QLoRA 微调 —— M3 任务 2，同时是前置件② 的载体。

## 这个脚本要证明两件事

1. **链路能跑通**：Transformers 加载 + bitsandbytes 4-bit 量化 + PEFT LoRA
   + Accelerate 训练循环，完整走一遍。这本身就是大纲「掌握模型微调核心
   算法能力」的落点。
2. **微调有没有效**：微调前 vs 微调后在 ScreenAgent 验证集上的差值
   （动作类型准确率 + 坐标误差），对应大纲 W5「对比微调前后模型在
   GUI 任务理解与动作生成上的效果」。

**「微调无显著提升」是一个合法结论**（M3 降级路径 2），只要训练过程
可复现、超参记录完整。实测训练集只有 569 条，远低于大纲定的 3000-4000，
所以这个结论是可预期的——**提前说清楚，比事后找补诚实**。

## 先跑 smoke test

    python -m finetune.train_lora --smoke

100 步、不保存权重，只回答一个问题：**这条链路在这台机器上跑不跑得起来。**
M3 D1 晚间要启动挂机训练，那时候才发现 bitsandbytes 装错了、显存不够、
或者 processor 的图片字段名对不上，就是白白浪费一夜。

前置件②要的就是这一步的输出。

## 显存的大头不是权重

3B 模型 4-bit 量化后权重约 2-3GB，但 Qwen2.5-VL 是动态分辨率——一张
2560×1600 截图会产生 2000+ 个视觉 token，视觉塔与注意力的激活显存可能
超过权重本身。

所以 `--max-pixels` 是必须调的参数，不是可选项。默认值按「最长边 896px」
折算，这是里程碑文档定的。**峰值显存以实测为准，报告里不写「约 3GB」
这类估算。**

## 精度按卡选

- 30 系及以上（Ampere+）：`bf16`
- T4 / V100：只能 `fp16`

选错的表现是 loss 变 NaN，而不是报错。`--precision auto` 会按算力自动选。

## 用法

    python -m finetune.train_lora --smoke                     # 100 步冒烟
    python -m finetune.train_lora                             # 完整训练
    python -m finetune.train_lora --epochs 2 --lr 1e-4
    python -m finetune.train_lora --model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("finetune/data")
OUT_DIR = Path("finetune/outputs")

#: 里程碑文档定的 PEFT 配置。改这里等于改实验，要同步改报告。
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
#: 覆盖注意力与 MLP 的全部投影层。只挂 q/v 是更省的做法，但对
#: 「学一个新的输出空间（坐标）」这类任务，MLP 也要动。
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: 最长边 896px 折算出的视觉 token 上限。28×28 是 Qwen2.5-VL 的 patch 尺寸。
DEFAULT_MAX_PIXELS = 896 * 896


@dataclass
class TrainStats:
    """训练过程的实测记录。**报告里一律用这里的值，不写估算。**"""

    model: str = ""
    smoke: bool = False
    steps: int = 0
    epochs: float = 0.0
    train_samples: int = 0
    val_samples: int = 0
    precision: str = ""
    max_pixels: int = 0
    peak_vram_mb: float = 0.0
    load_seconds: float = 0.0
    train_seconds: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    losses: list = field(default_factory=list)
    error: str = ""

    @property
    def trainable_ratio(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0


def pick_precision(explicit: str = "auto") -> str:
    """按显卡算力选精度。

    **选错不会报错，只会让 loss 变 NaN。** Ampere 之前的卡没有 bf16 的
    硬件支持，跑起来是软件模拟或直接出错。
    """
    if explicit != "auto":
        return explicit
    try:
        import torch

        if not torch.cuda.is_available():
            return "fp32"
        major, _ = torch.cuda.get_device_capability()
        return "bf16" if major >= 8 else "fp16"
    except Exception:  # noqa: BLE001
        return "fp16"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


#: 训练用的系统提示词。**与 `prompts/executor_v1.yaml` 的动作规范同构。**
#:
#: 训练时不给动作空间、推理时给一大段动作说明，模型见到的分布就是两回事。
#: 这里不照抄 executor_v1 的全文（那份还带 few-shot 与子任务约定，对单步
#: 动作生成是噪声），只保留决定输出格式的那部分：动作名与 JSON 结构。
#:
#: 坐标空间与 `finetune.dataset.COORD_SPACE` 一致，不在提示词里另写一套。
SYSTEM_PROMPT = """你是一个桌面 GUI 智能体。你会看到一张屏幕截图和一个任务目标，
需要判断为完成该任务，下一步应当执行什么动作。

可用动作：left_click / double_click / right_click / mouse_move

只输出一个 JSON 对象，不要任何解释文字：

{"action": "动作名", "x": 横坐标, "y": 纵坐标}

坐标按 1000×1000 的范围给出，原点在左上角。"""


def build_messages(record: dict) -> list[dict]:
    """一条样本的对话形式。

    **与推理时的构造保持同构。** 训练时如果把图片放在 user 消息里、
    推理时放在 system 里，或者训练时不给动作空间而推理时给，模型见到的
    分布就不是一回事——**而这种错误在 loss 曲线上完全看不出来**，
    只会让评测分数莫名其妙地低。
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["image"]},
                {"type": "text", "text": record["instruction"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]},
    ]


def train(args) -> TrainStats:  # noqa: PLR0915 —— 训练脚本，线性叙事好读
    stats = TrainStats(model=args.model, smoke=args.smoke)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    train_path = Path(args.data) / "train.jsonl"
    val_path = Path(args.data) / "val.jsonl"
    if not train_path.exists():
        raise SystemExit(f"找不到 {train_path}。先跑 python -m finetune.dataset")

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path.exists() else []
    if args.smoke:
        # 冒烟只要够跑满 100 步，不需要全量
        train_rows = train_rows[: max(args.max_steps * args.grad_accum, 32)]
        val_rows = val_rows[:8]
    stats.train_samples = len(train_rows)
    stats.val_samples = len(val_rows)

    stats.precision = pick_precision(args.precision)
    stats.max_pixels = args.max_pixels
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(stats.precision, torch.float32)

    print(f"  精度 {stats.precision}    max_pixels {args.max_pixels}")
    print(f"  训练 {stats.train_samples} 条    验证 {stats.val_samples} 条")

    load_started = time.perf_counter()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quant, dtype=dtype, device_map="auto"
    )
    stats.load_seconds = round(time.perf_counter() - load_started, 1)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # 与 gradient checkpointing 冲突
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stats.trainable_params, stats.total_params = trainable, total
    print(f"  可训练参数 {trainable:,} / {total:,} = {trainable / total:.4%}")

    def collate(batch: list[dict]) -> dict:
        """把一批样本编码成模型输入。

        **只对 assistant 的回答算 loss。** 不 mask 掉 prompt 的话，
        模型会把大部分梯度花在复述指令上，而我们要它学的只有坐标。
        """
        from qwen_vl_utils import process_vision_info

        texts, images = [], []
        for record in batch:
            messages = build_messages(record)
            texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
            image_inputs, _ = process_vision_info(messages)
            images.append(image_inputs)

        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True, truncation=False
        )
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for index, record in enumerate(batch):
            answer_ids = processor.tokenizer(record["answer"], add_special_tokens=False)[
                "input_ids"
            ]
            # 回答在序列末尾，把它之前的全部 mask 掉
            keep = len(answer_ids)
            if keep < labels.shape[1]:
                labels[index, : labels.shape[1] - keep] = -100
        inputs["labels"] = labels
        return inputs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / ("smoke" if args.smoke else time.strftime("%Y%m%d-%H%M%S"))

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.smoke else -1,
        learning_rate=args.lr,
        # paged_adamw_8bit：优化器状态分页到内存，显存吃紧时不 OOM
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if args.smoke else 10,
        # 每 200 步存一次，训练中断不至于从头再来
        save_steps=10**9 if args.smoke else 200,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=stats.precision == "bf16",
        fp16=stats.precision == "fp16",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        data_collator=collate,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_started = time.perf_counter()
    try:
        result = trainer.train()
        stats.steps = int(result.global_step)
        stats.epochs = round(float(result.metrics.get("epoch", 0.0)), 3)
    except Exception as exc:  # noqa: BLE001 —— 失败也要留下实测记录
        stats.error = f"{type(exc).__name__}: {exc}"[:400]
        raise
    finally:
        stats.train_seconds = round(time.perf_counter() - train_started, 1)
        if torch.cuda.is_available():
            stats.peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        stats.losses = [
            {"step": entry.get("step"), "loss": entry.get("loss")}
            for entry in trainer.state.log_history
            if "loss" in entry
        ]

    if not args.smoke:
        model.save_pretrained(str(run_dir / "adapter"))
        processor.save_pretrained(str(run_dir / "adapter"))
        print(f"  adapter 已保存到 {run_dir / 'adapter'}")
    else:
        print("  冒烟模式：**不保存权重**，只验证链路。")

    (run_dir / "train-stats.json").write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B QLoRA 微调（M3 任务 2）")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--smoke", action="store_true", help="100 步冒烟，不保存权重")
    parser.add_argument("--max-steps", type=int, default=100, help="冒烟模式的步数")
    args = parser.parse_args()

    print("=" * 70)
    print("QLoRA 微调" + ("（冒烟）" if args.smoke else ""))
    print("=" * 70)
    print(f"  模型  {args.model}")
    print(f"  LoRA  r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
    print(f"  目标层 {', '.join(TARGET_MODULES)}")
    print()

    stats = train(args)

    print()
    print("=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"  步数        {stats.steps}    epoch {stats.epochs}")
    print(f"  加载耗时    {stats.load_seconds}s")
    print(f"  训练耗时    {stats.train_seconds}s")
    print(f"  峰值显存    **{stats.peak_vram_mb} MB**  ← 报告里用这个实测值")
    print(f"  可训练参数  {stats.trainable_params:,}（{stats.trainable_ratio:.4%}）")
    if stats.losses:
        first, last = stats.losses[0], stats.losses[-1]
        print(f"  loss        {first['loss']:.4f} → {last['loss']:.4f}")
    if args.smoke:
        print("\n  **前置件② 完成。** 链路可跑通，把上面的峰值显存记进 M3 计划，")
        print("  D1 下午的本地部署直接沿用这个 max_pixels。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
