"""Inspect packaged USD collision meshes after launching Isaac Sim.

This diagnostic is intentionally read-only.  It reports the prim paths and
collision schemas needed when authoring the hybrid primitive/SDF assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("paths", nargs="*", type=Path, help="USD files to inspect")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    REPO / "source/bin_pickplace_tasks/bin_pickplace_tasks/assets/A00038 Screwdriver-IsaacSim/screwdriver_physics.usd",
    REPO / "source/bin_pickplace_tasks/bin_pickplace_tasks/assets/A00039 Wrench-IsaacSim/wrench_physics.usd",
    REPO / "source/bin_pickplace_tasks/bin_pickplace_tasks/assets/A00040 Hammer-IsaacSim/hammer_physics.usd",
    REPO
    / "source/bin_pickplace_tasks/bin_pickplace_tasks/assets/N00838 Sm Paint Supplies Palletknife Simready-IsaacSim/pallet_knife_physics.usd",
)


def inspect(path: Path) -> None:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    print(f"\n=== {path} ===", flush=True)
    print(f"defaultPrim={stage.GetDefaultPrim().GetPath()}", flush=True)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.TraverseAll():
        is_mesh = prim.IsA(UsdGeom.Mesh)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_sdf = prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
        if not (is_mesh or has_collision or has_rigid or has_sdf):
            continue
        fields = [f"path={prim.GetPath()}", f"type={prim.GetTypeName()}"]
        if prim.IsInstance() or prim.IsInstanceable():
            fields.append(f"instance={prim.IsInstance()}/{prim.IsInstanceable()}")
        if is_mesh:
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            faces = mesh.GetFaceVertexCountsAttr().Get() or []
            bounds = cache.ComputeLocalBound(prim).ComputeAlignedRange()
            geom_subsets = [child for child in prim.GetChildren() if child.GetTypeName() == "GeomSubset"]
            fields.extend(
                [
                    f"points={len(points)}",
                    f"faces={len(faces)}",
                    f"geom_subsets={len(geom_subsets)}",
                    f"bounds=({tuple(round(v, 6) for v in bounds.GetMin())},"
                    f"{tuple(round(v, 6) for v in bounds.GetMax())})",
                ]
            )
        if has_collision:
            fields.append("collision=True")
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            fields.append(f"approximation={approximation}")
        if has_sdf:
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI(prim)
            fields.extend(
                [
                    f"sdf_resolution={sdf.GetSdfResolutionAttr().Get()}",
                    f"sdf_subgrid={sdf.GetSdfSubgridResolutionAttr().Get()}",
                ]
            )
        if has_rigid:
            fields.append("rigid=True")
        print("  " + " | ".join(fields), flush=True)
        if is_mesh and "screwdriver" in path.name:
            array = np.asarray(points, dtype=np.float64)
            x_min, x_max = array[:, 0].min(), array[:, 0].max()
            print("    screwdriver x-profile (occupied local y/z bounds):", flush=True)
            for left, right in zip(np.linspace(x_min, x_max, 17)[:-1], np.linspace(x_min, x_max, 17)[1:]):
                selected = array[(array[:, 0] >= left) & (array[:, 0] <= right)]
                if not len(selected):
                    continue
                print(
                    f"      x=[{left:+.4f},{right:+.4f}] n={len(selected):5d} "
                    f"y=[{selected[:, 1].min():+.4f},{selected[:, 1].max():+.4f}] "
                    f"z=[{selected[:, 2].min():+.4f},{selected[:, 2].max():+.4f}]",
                    flush=True,
                )


try:
    for asset_path in args.paths or DEFAULT_PATHS:
        inspect(asset_path)
finally:
    simulation_app.close()
