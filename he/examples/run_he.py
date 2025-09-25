import argparse
from pathlib import Path
from pprint import pprint
from sdk_he import load_config, SecureHEPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    # 把传入的路径规整成绝对路径（相对当前工作目录）
    cfg_path = str(Path(args.config).expanduser().resolve())
    print(f"Using config: {cfg_path}")

    cfg = load_config(cfg_path)
    pipe = SecureHEPipeline(cfg)
    result = pipe.run_all()

    print("\n===== Training & Evaluation Finished =====")
    print(f"Classes: {result['classes']}")
    print(f"Num features: {len(result['feature_names'])}")
    print("Metrics summary:")
    pprint(result["metrics"])

if __name__ == "__main__":
    main()