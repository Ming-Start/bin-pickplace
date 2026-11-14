"""Normalize converted bin USDs and remove development-machine texture paths.

Run with Blender's bundled Python/pxr runtime:
    blender --background --python scripts/prepare_portable_assets.py

The two input USDs are intermediate products of Isaac Lab's MeshConverter and are kept in
``/tmp`` only while preparing the repository.  The final USDs are self-contained relative to
the package asset directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "source/bin_pickplace_tasks/bin_pickplace_tasks/assets"


def make_paths_relative(stage: Usd.Stage, source_prefix: str, relative_prefix: str = "textures") -> None:
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and value.path.startswith(source_prefix):
                attr.Set(Sdf.AssetPath(f"{relative_prefix}/{Path(value.path).name}"))


def rewrite_exported_paths(path: Path) -> None:
    """Author relative paths after export so USD does not preserve the old resolved anchor."""
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not reopen exported USD: {path}")
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and value.path.startswith("/tmp/") and "/textures/" in value.path:
                attr.Set(Sdf.AssetPath(f"textures/{Path(value.path).name}"))
    stage.GetRootLayer().Save()


def prepare_blue() -> None:
    stage = Usd.Stage.Open("/tmp/blue_bin_glb_physics.usd")
    if stage is None:
        raise RuntimeError("Missing /tmp/blue_bin_glb_physics.usd")

    # The supplied GLB is Y-up with its rigid origin above the geometric bottom.  Rotate its
    # child geometry into the task's Z-up convention and shift the bottom to local z=0 while
    # keeping the rigid-body root itself at the bin-floor reference.
    geometry = stage.GetPrimAtPath("/textured_parts/geometry")
    xform = UsdGeom.Xformable(geometry)
    xform.ClearXformOpOrder()
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90.0))
    transform.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.1917818784713745))
    xform.AddTransformOp().Set(transform)
    make_paths_relative(stage, "/tmp/textures/")

    target_dir = ASSETS / "A03071 Blue Stackable Plastic Bin 40L"
    shutil.copy2("/tmp/textures/textured_parts_texture0.png", target_dir / "textures/textured_parts_texture0.png")
    shutil.copy2("/tmp/textures/textured_parts_texture1.png", target_dir / "textures/textured_parts_texture1.png")
    target = target_dir / "blue_bin_physics.usd"
    stage.Export(str(target))
    rewrite_exported_paths(target)


def prepare_white() -> None:
    stage = Usd.Stage.Open("/tmp/white_bin_glb_physics.usd")
    if stage is None:
        raise RuntimeError("Missing /tmp/white_bin_glb_physics.usd")
    make_paths_relative(stage, "/tmp/textures/")
    target_dir = ASSETS / "N01705 Bin-IsaacSim"
    target = target_dir / "white_bin_physics.usd"
    stage.Export(str(target))
    rewrite_exported_paths(target)


def prepare_tools() -> None:
    specs = {
        "screwdriver": ("A00038 Screwdriver-IsaacSim", "screwdriver_physics.usd"),
        "wrench": ("A00039 Wrench-IsaacSim", "wrench_physics.usd"),
        "hammer": ("A00040 Hammer-IsaacSim", "hammer_physics.usd"),
        "pallet_knife": (
            "N00838 Sm Paint Supplies Palletknife Simready-IsaacSim",
            "pallet_knife_physics.usd",
        ),
    }
    for build_name, (asset_folder, filename) in specs.items():
        build_dir = Path("/tmp/bin_pickplace_assets") / build_name
        source = build_dir / filename
        stage = Usd.Stage.Open(str(source))
        if stage is None:
            raise RuntimeError(f"Missing converted tool USD: {source}")
        target_dir = ASSETS / asset_folder
        target = target_dir / filename
        stage.Export(str(target))
        texture_source = build_dir / "textures"
        if texture_source.exists():
            for texture in texture_source.iterdir():
                if texture.is_file():
                    shutil.copy2(texture, target_dir / "textures" / texture.name)
        rewrite_exported_paths(target)


prepare_blue()
prepare_white()
prepare_tools()
print("Prepared six portable physics USD assets")
