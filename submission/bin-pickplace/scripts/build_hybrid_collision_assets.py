"""Build portable hybrid primitive/SDF collision variants for Bin-PickPlace tools.

The original packaged USD files are never overwritten.  This script preserves
their visual meshes and materials while authoring two new runtime assets:

* screwdriver_hybrid_physics.usd: capsule shaft + convex-hull handle
* wrench_sdf_physics.usd: sparse SDF on the supplied single visual mesh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "source/bin_pickplace_tasks/bin_pickplace_tasks/assets"
SCREWDRIVER_DIR = ASSETS / "A00038 Screwdriver-IsaacSim"
WRENCH_DIR = ASSETS / "A00039 Wrench-IsaacSim"

SCREWDRIVER_SOURCE = SCREWDRIVER_DIR / "screwdriver_physics.usd"
SCREWDRIVER_TARGET = SCREWDRIVER_DIR / "screwdriver_hybrid_physics.usd"
WRENCH_SOURCE = WRENCH_DIR / "wrench_physics.usd"
WRENCH_TARGET = WRENCH_DIR / "wrench_sdf_physics.usd"

# Measured directly from the packaged screwdriver mesh.  The shaft occupies
# local x [-0.1417, -0.0420] around z=0.0157; the handle starts at x~-0.045.
SHAFT_CENTER = (-0.08935, 0.0, 0.01570)
SHAFT_RADIUS = 0.00370
SHAFT_CAPSULE_HEIGHT = 0.09730
HANDLE_FACE_CENTROID_X_MIN = -0.045

WRENCH_SDF_RESOLUTION = 256
WRENCH_SDF_SUBGRID_RESOLUTION = 6


def remove_mesh_collision(prim: Usd.Prim) -> None:
    """Remove collider schemas while leaving the visual mesh untouched."""
    for schema in (
        PhysxSchema.PhysxSDFMeshCollisionAPI,
        PhysxSchema.PhysxCollisionAPI,
        UsdPhysics.MeshCollisionAPI,
        UsdPhysics.CollisionAPI,
    ):
        if prim.HasAPI(schema):
            prim.RemoveAPI(schema)


def build_handle_convex(source_mesh: UsdGeom.Mesh, target_path: str) -> UsdGeom.Mesh:
    """Extract handle faces and author a hidden convex-hull collision mesh."""
    points = list(source_mesh.GetPointsAttr().Get() or [])
    face_counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
    face_indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
    selected_faces: list[list[int]] = []
    cursor = 0
    for count in face_counts:
        indices = face_indices[cursor : cursor + count]
        cursor += count
        centroid_x = sum(float(points[index][0]) for index in indices) / count
        if centroid_x >= HANDLE_FACE_CENTROID_X_MIN:
            selected_faces.append(indices)
    if not selected_faces:
        raise RuntimeError("No screwdriver handle faces were selected")

    used_indices = sorted({index for face in selected_faces for index in face})
    index_map = {old: new for new, old in enumerate(used_indices)}
    handle_points = [points[index] for index in used_indices]
    handle_counts = [len(face) for face in selected_faces]
    handle_indices = [index_map[index] for face in selected_faces for index in face]

    target = UsdGeom.Mesh.Define(source_mesh.GetPrim().GetStage(), target_path)
    target.CreatePointsAttr(handle_points)
    target.CreateFaceVertexCountsAttr(handle_counts)
    target.CreateFaceVertexIndicesAttr(handle_indices)
    target.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    array_min = Gf.Vec3f(
        min(point[0] for point in handle_points),
        min(point[1] for point in handle_points),
        min(point[2] for point in handle_points),
    )
    array_max = Gf.Vec3f(
        max(point[0] for point in handle_points),
        max(point[1] for point in handle_points),
        max(point[2] for point in handle_points),
    )
    target.CreateExtentAttr([array_min, array_max])
    target.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(target.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(target.GetPrim())
    mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    return target


def build_screwdriver() -> None:
    stage = Usd.Stage.Open(str(SCREWDRIVER_SOURCE))
    if stage is None:
        raise RuntimeError(f"Cannot open {SCREWDRIVER_SOURCE}")
    root = stage.GetDefaultPrim()
    visual_prim = stage.GetPrimAtPath("/screwdriver/geometry/mesh")
    if not visual_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("Expected screwdriver mesh at /screwdriver/geometry/mesh")
    remove_mesh_collision(visual_prim)

    collider_root = UsdGeom.Xform.Define(stage, "/screwdriver/hybrid_colliders")
    collider_root.GetPrim().SetCustomDataByKey("binPickplaceRepresentation", "capsule shaft + convex handle")
    shaft = UsdGeom.Capsule.Define(stage, "/screwdriver/hybrid_colliders/shaft_capsule")
    shaft.CreateAxisAttr(UsdGeom.Tokens.x)
    shaft.CreateRadiusAttr(SHAFT_RADIUS)
    shaft.CreateHeightAttr(SHAFT_CAPSULE_HEIGHT)
    shaft.AddTranslateOp().Set(Gf.Vec3f(*SHAFT_CENTER))
    shaft.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())

    handle = build_handle_convex(
        UsdGeom.Mesh(visual_prim), "/screwdriver/hybrid_colliders/handle_convex"
    )
    root.SetCustomDataByKey("binPickplaceCollisionAsset", "hybrid-v1")
    root.SetCustomDataByKey("binPickplaceShaftRadiusM", SHAFT_RADIUS)
    root.SetCustomDataByKey("binPickplaceHandleFaceThresholdX", HANDLE_FACE_CENTROID_X_MIN)
    stage.Export(str(SCREWDRIVER_TARGET))
    print(
        f"Built {SCREWDRIVER_TARGET} with shaft capsule and "
        f"{len(handle.GetFaceVertexCountsAttr().Get())} handle faces",
        flush=True,
    )


def build_wrench() -> None:
    stage = Usd.Stage.Open(str(WRENCH_SOURCE))
    if stage is None:
        raise RuntimeError(f"Cannot open {WRENCH_SOURCE}")
    root = stage.GetDefaultPrim()
    mesh_prim = stage.GetPrimAtPath("/wrench/geometry/mesh")
    if not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("Expected wrench mesh at /wrench/geometry/mesh")
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr(PhysxSchema.Tokens.sdf)
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
    sdf.CreateSdfResolutionAttr(WRENCH_SDF_RESOLUTION)
    sdf.CreateSdfSubgridResolutionAttr(WRENCH_SDF_SUBGRID_RESOLUTION)
    root.SetCustomDataByKey("binPickplaceCollisionAsset", "sparse-sdf-v1")
    root.SetCustomDataByKey("binPickplaceSdfResolution", WRENCH_SDF_RESOLUTION)
    root.SetCustomDataByKey("binPickplaceSdfSubgridResolution", WRENCH_SDF_SUBGRID_RESOLUTION)
    stage.Export(str(WRENCH_TARGET))
    print(
        f"Built {WRENCH_TARGET} with sparse SDF "
        f"{WRENCH_SDF_RESOLUTION}/{WRENCH_SDF_SUBGRID_RESOLUTION}",
        flush=True,
    )


def validate_outputs() -> None:
    screwdriver = Usd.Stage.Open(str(SCREWDRIVER_TARGET))
    wrench = Usd.Stage.Open(str(WRENCH_TARGET))
    if screwdriver is None or wrench is None:
        raise RuntimeError("Could not reopen generated hybrid assets")
    old_visual = screwdriver.GetPrimAtPath("/screwdriver/geometry/mesh")
    shaft = screwdriver.GetPrimAtPath("/screwdriver/hybrid_colliders/shaft_capsule")
    handle = screwdriver.GetPrimAtPath("/screwdriver/hybrid_colliders/handle_convex")
    if old_visual.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Original screwdriver visual mesh still has CollisionAPI")
    if not shaft.HasAPI(UsdPhysics.CollisionAPI) or not handle.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Generated screwdriver colliders are incomplete")
    if UsdPhysics.MeshCollisionAPI(handle).GetApproximationAttr().Get() != UsdPhysics.Tokens.convexHull:
        raise RuntimeError("Screwdriver handle is not convexHull")
    wrench_mesh = wrench.GetPrimAtPath("/wrench/geometry/mesh")
    if UsdPhysics.MeshCollisionAPI(wrench_mesh).GetApproximationAttr().Get() != PhysxSchema.Tokens.sdf:
        raise RuntimeError("Wrench approximation is not SDF")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI(wrench_mesh)
    if sdf.GetSdfResolutionAttr().Get() != WRENCH_SDF_RESOLUTION:
        raise RuntimeError("Wrench SDF resolution mismatch")
    if sdf.GetSdfSubgridResolutionAttr().Get() != WRENCH_SDF_SUBGRID_RESOLUTION:
        raise RuntimeError("Wrench sparse SDF subgrid mismatch")


try:
    build_screwdriver()
    build_wrench()
    validate_outputs()
    print("Hybrid collision assets validated", flush=True)
finally:
    simulation_app.close()
