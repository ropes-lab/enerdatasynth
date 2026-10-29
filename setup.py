"""Copies the shipped default checkpoints into the wheel.

All metadata lives in pyproject.toml. This file only adds one build step:
the files listed in BUNDLED below (the shipped Transformer for each of the 17
PJM zones with its scaler file and calibration, and the three wind presets) are copied from the repository's
data/ folder into `enerdatasynth/_bundled/` inside the built package, so a
plain `pip install enerdatasynth` can generate without a repo checkout.
`enerdatasynth.utils.config` reads them from there when the project folder
has no copy of its own. Keep this list in sync with MANIFEST.in.
"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent

RUN_ID = "transformer_T1000_H64_L2"
# One trained model per PJM zone. PS keeps the original flat folders; every other
# zone lives in zones/<ZONE>/ next to them. Keep in sync with PJM_ZONES in
# src/enerdatasynth/utils/config.py.
PJM_ZONES = ["PS", "AE", "AP", "BC", "CE", "DAY", "DEOK", "DOM", "DPL", "DUQ",
             "EKPC", "JC", "ME", "PE", "PL", "PN", "RECO"]


def _zone_dir(category: str, zone: str) -> str:
    base = f"{category}/pjm_load"
    return base if zone == "PS" else f"{base}/zones/{zone}"


# (source under data/, destination under enerdatasynth/_bundled/)
BUNDLED = [
    *[
        (f"{_zone_dir('processed', zone)}/metadata.json", f"{_zone_dir('processed', zone)}/metadata.json")
        for zone in PJM_ZONES
    ],
    *[
        (f"{_zone_dir('checkpoints', zone)}/{RUN_ID}/{name}", f"{_zone_dir('checkpoints', zone)}/{RUN_ID}/{name}")
        for zone in PJM_ZONES
        for name in ("best.pt", "result.json", "calibration.json")
    ],
    *[
        (f"checkpoints/wind/{preset}/{name}", f"checkpoints/wind/{preset}/{name}")
        for preset in ("1ch", "2ch", "3ch")
        for name in ("model_checkpoint.pt", "norm_stats.json")
    ],
]


class BuildPyWithBundledData(build_py):
    def run(self):
        super().run()
        if self.dry_run:
            return
        target_root = Path(self.build_lib) / "enerdatasynth" / "_bundled"
        for src_rel, dst_rel in BUNDLED:
            src = HERE / "data" / src_rel
            if not src.exists():
                raise FileNotFoundError(
                    f"Cannot build the wheel: {src} is missing. These files are "
                    f"shipped inside the package; see setup.py."
                )
            dst = target_root / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


setup(cmdclass={"build_py": BuildPyWithBundledData})
