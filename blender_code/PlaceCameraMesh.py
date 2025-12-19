import bpy
import math
from mathutils import Matrix, Vector, Euler

# ----------------------------
# SETTINGS YOU MAY TWEAK
# ----------------------------
CAMERA_NAMES = [f"GoPro{i}" for i in range(1, 13)]

MODEL_COLLECTION_NAME = "GoPro_Model"   # collection that contains the imported GLB objects
INSTANCES_COLLECTION_NAME = "GoPro_Instances"
HIDE_OLD_MARKERS = True                 # hides objects named GoProX_Marker if they exist

# Orientation offset applied to the model so it matches the camera.
# Common good guess: many models are "forward +Y, up +Z" -> rotate -90° around X to make forward -Z, up +Y
ROT_OFFSET_DEG = (90.0, 0.0, 180.0)      # (X, Y, Z) degrees

# Optional fine adjustment (local translation after rotation), e.g. move model so lens sits exactly at camera origin
LOC_OFFSET = (0.0, 0.0, 0.0)            # meters in the model's local space

# Optional scale factor (if the GLB is huge/tiny). 1.0 = keep as imported.
SCALE = 0.1# ----------------------------


def ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


model_col = bpy.data.collections.get(MODEL_COLLECTION_NAME)
if model_col is None:
    raise RuntimeError(f"Collection '{MODEL_COLLECTION_NAME}' not found. Put your imported GoPro objects into that collection.")

instances_col = ensure_collection(INSTANCES_COLLECTION_NAME)

# Build offset transform (camera space -> model space)
rot = Euler(tuple(math.radians(a) for a in ROT_OFFSET_DEG), 'XYZ').to_matrix().to_4x4()
loc = Matrix.Translation(Vector(LOC_OFFSET))
scl = Matrix.Diagonal((SCALE, SCALE, SCALE, 1.0))
offset_mtx = loc @ rot @ scl

missing = []
created_or_updated = 0

for cam_name in CAMERA_NAMES:
    cam = bpy.data.objects.get(cam_name)
    if cam is None or cam.type != "CAMERA":
        missing.append(cam_name)
        continue

    inst_name = f"{cam_name}_GoProModel"

    # Hide old marker if present
    if HIDE_OLD_MARKERS:
        m = bpy.data.objects.get(f"{cam_name}_Marker")
        if m:
            m.hide_viewport = True
            m.hide_render = True

    inst = bpy.data.objects.get(inst_name)
    if inst is None:
        inst = bpy.data.objects.new(inst_name, None)
        inst.instance_type = 'COLLECTION'
        inst.instance_collection = model_col
        instances_col.objects.link(inst)

    # Place instance at camera pose (plus offset)
    inst.matrix_world = cam.matrix_world @ offset_mtx

    # Parent to camera (so it follows camera moves), keep transform
    inst.parent = cam
    inst.matrix_parent_inverse = cam.matrix_world.inverted()

    created_or_updated += 1

print(f"[GoPro Replace] Created/updated {created_or_updated} GoPro instances.")
if missing:
    print("[GoPro Replace] Missing cameras (or not CAMERA type):", missing)
