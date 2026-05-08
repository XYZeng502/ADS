import argparse
import json
from pathlib import Path

from app.schemas import ClientContext
from app.services.mpc_snapshot import MPCSnapshotService
from app.services.orchestrator import DailyOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="运行每日预算决策示例")
    parser.add_argument("--input", default="", help="输入 JSON，默认使用 sample_input.json")
    parser.add_argument("--write-mpc-snapshot", action="store_true", help="写入 MPC 决策快照，供次日校准")
    parser.add_argument("--snapshot-output-dir", default="outputs/mpc_snapshots", help="MPC 快照输出目录")
    args = parser.parse_args()

    sample_file = Path(args.input) if args.input else Path(__file__).resolve().parents[1] / "sample_input.json"
    context_data = json.loads(sample_file.read_text(encoding="utf-8"))
    context = ClientContext(**context_data)
    result = DailyOrchestrator().run(context)
    if args.write_mpc_snapshot:
        snapshot = MPCSnapshotService().save(context=context, plan=result, output_dir=args.snapshot_output_dir)
        result.notes.append(f"mpc_snapshot_path={snapshot.snapshot_path}")
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
