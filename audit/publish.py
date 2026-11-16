import argparse
import shutil
import subprocess
from pathlib import Path

from audit.common import load_config, output_path
from audit.figures import check_figure_links, figure_links


def build_pdf(cfg):
    check_figure_links(cfg)
    html = output_path(cfg, "rapport.html")
    pdf = output_path(cfg, cfg["publish"]["pdf"])
    subprocess.run(
        [
            "pandoc",
            cfg["paths"]["methodology"],
            "-s",
            "--metadata",
            f"pagetitle={cfg['publish']['title']}",
            "-o",
            str(html),
        ],
        check=True,
    )
    css = Path(__file__).parent / "rapport.css"
    subprocess.run(
        [
            "weasyprint",
            "--optimize-images",
            "--dpi",
            str(cfg["figures"]["dpi"]),
            "--jpeg-quality",
            str(cfg["figures"]["jpeg_quality"]),
            "-s",
            str(css),
            str(html),
            str(pdf),
        ],
        check=True,
    )
    return pdf


def copy_to_targets(cfg, pdf):
    methodology = Path(cfg["paths"]["methodology"])
    for target in cfg["publish"]["targets"]:
        out = Path(target["dir"])
        (out / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, out / pdf.name)
        if target["markdown"]:
            shutil.copy2(methodology, out / target["markdown"])
            names = figure_links(methodology.read_text())
        else:
            figures = output_path(cfg, "figures")
            names = [
                f"figures/{path.name}"
                for pattern in target["figures"]
                for path in sorted(figures.glob(pattern))
            ]
        for name in names:
            shutil.copy2(output_path(cfg, name), out / name)
        print(f"-> {out}")


def main():
    parser = argparse.ArgumentParser(description="PDF et copie vers docs et site")
    parser.add_argument("--config", default="audit/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pdf = build_pdf(cfg)
    print(f"pdf -> {pdf}")
    copy_to_targets(cfg, pdf)


if __name__ == "__main__":
    main()
