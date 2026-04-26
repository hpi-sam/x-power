import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VehicleTypeConfig:
    """Parameters for a single vehicle type."""
    vtype_id        : str
    speed           : float = 13.89         # m/s (~50 km/h default)
    accel           : float = 2.6
    decel           : float = 4.5
    sigma           : float = 0.5           # driver imperfection [0,1]
    length          : float = 5.0           # metres
    min_gap         : float = 2.5           # metres
    friction_coeff  : float = 1.0           # tyre-road friction [0,1]
    color           : str   = "255,255,0"   # RGB for GUI
    # SSM (Safety Surrogate Measures) device
    ssm_measures    : list  = field(default_factory=lambda: ["TTC", "DRAC", "PET"])
    ssm_thresholds  : list  = field(default_factory=lambda: [3.0, 3.0, 2.0])
    ssm_range       : float = 50.0          # detection range in metres


# ── Preset vehicle types ───────────────────────────────────────────────────────
VEHICLE_PRESETS = {
    "standard": VehicleTypeConfig(
        vtype_id="standard",
        speed=13.89, accel=2.6, decel=4.5,
        sigma=0.5, friction_coeff=1.0, color="255,255,0"
    ),
    "aggressive": VehicleTypeConfig(
        vtype_id="aggressive",
        speed=22.0, accel=4.0, decel=6.0,
        sigma=0.8, friction_coeff=1.0, color="255,0,0"
    ),
    "cautious": VehicleTypeConfig(
        vtype_id="cautious",
        speed=8.33, accel=1.5, decel=3.0,
        sigma=0.1, friction_coeff=1.0, color="0,0,255"
    ),
    "slippery": VehicleTypeConfig(
        vtype_id="slippery",
        speed=13.89, accel=2.6, decel=2.0,   # reduced decel = slippery road
        sigma=0.5, friction_coeff=0.4, color="0,255,255"
    ),
    "heavy": VehicleTypeConfig(
        vtype_id="heavy",
        speed=8.33, accel=1.0, decel=3.0,
        sigma=0.3, length=12.0, min_gap=3.0,
        friction_coeff=0.9, color="128,64,0"
    ),
}


class TrafficConfig:
    """
    Builds SUMO additional files (.add.xml) defining vehicle types and
    SSM devices. These are passed to SumoEnvironment via additional_sumo_cmd.

    Usage
    -----
    cfg = TrafficConfig(output_dir="nets/my_net")

    # Homogeneous — all same type
    cfg.add_preset("aggressive")

    # Heterogeneous — mix of types
    cfg.add_preset("standard")
    cfg.add_preset("cautious")
    cfg.add_custom(VehicleTypeConfig(vtype_id="my_truck", length=15, speed=8))

    add_file = cfg.write()   # writes the .add.xml and returns its path
    # Pass to env:
    additional_sumo_cmd = ["--additional-files", add_file]
    """

    def __init__(self, output_dir: str):
        self.output_dir  = Path(output_dir)
        self.vtypes      : list[VehicleTypeConfig] = []

    def add_preset(self, name: str) -> "TrafficConfig":
        """Add a named preset vehicle type."""
        assert name in VEHICLE_PRESETS, \
            f"Unknown preset '{name}'. Choose from: {list(VEHICLE_PRESETS.keys())}"
        self.vtypes.append(VEHICLE_PRESETS[name])
        return self

    def add_custom(self, vtype: VehicleTypeConfig) -> "TrafficConfig":
        """Add a fully custom vehicle type."""
        self.vtypes.append(vtype)
        return self

    def write(self, filename: str = "vehicle_types.add.xml") -> str:
        """
        Write the additional file to disk and return its absolute path (str).
        Call this BEFORE creating the SumoEnvironment.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        root = ET.Element("additional")

        for vt in self.vtypes:
            # ── vType element ──────────────────────────────────────────────────
            vtype_el = ET.SubElement(root, "vType", attrib={
                "id"         : vt.vtype_id,
                "maxSpeed"   : str(vt.speed),
                "accel"      : str(vt.accel),
                "decel"      : str(vt.decel),
                "sigma"      : str(vt.sigma),
                "length"     : str(vt.length),
                "minGap"     : str(vt.min_gap),
                "color"      : vt.color,
            })

            # ── Friction param ─────────────────────────────────────────────────
            ET.SubElement(vtype_el, "param", attrib={
                "key"  : "frictionCoefficient",
                "value": str(vt.friction_coeff),
            })

            # ── SSM device ────────────────────────────────────────────────────
            if vt.ssm_measures:
                ET.SubElement(vtype_el, "param", attrib={
                    "key"  : "has.ssm.device",
                    "value": "true",
                })
                ET.SubElement(vtype_el, "param", attrib={
                    "key"  : "device.ssm.measures",
                    "value": " ".join(vt.ssm_measures),
                })
                ET.SubElement(vtype_el, "param", attrib={
                    "key"  : "device.ssm.thresholds",
                    "value": " ".join(str(t) for t in vt.ssm_thresholds),
                })
                ET.SubElement(vtype_el, "param", attrib={
                    "key"  : "device.ssm.range",
                    "value": str(vt.ssm_range),
                })

        out_path = self.output_dir / filename
        tree     = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(out_path), xml_declaration=True, encoding="utf-8")

        print(f"✓ Traffic config written → {out_path}")
        return str(out_path)

    def summary(self):
        print(f"\nTraffic config — {len(self.vtypes)} vehicle type(s):")
        for vt in self.vtypes:
            print(f"  [{vt.vtype_id}] speed={vt.speed} m/s | "
                  f"friction={vt.friction_coeff} | "
                  f"sigma={vt.sigma} | "
                  f"SSM={vt.ssm_measures}")