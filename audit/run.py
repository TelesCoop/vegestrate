import argparse
import subprocess
import sys

STEPS = ["prepare", "compute", "examples", "report"]


def main():
    parser = argparse.ArgumentParser(description="Audit EVA 2023 / Vegestrate 2023")
    parser.add_argument("--config", default="audit/config.yaml")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "audit.selfcheck"], check=True)
    for step in args.steps:
        print(f"== {step}", flush=True)
        subprocess.run(
            [sys.executable, "-m", f"audit.{step}", "--config", args.config], check=True
        )


if __name__ == "__main__":
    main()
