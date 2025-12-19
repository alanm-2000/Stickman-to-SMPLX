import json
import os
import bpy
from mathutils import Matrix

# =================== EDIT THESE IF NEEDED ===================
JSON_RT_PATH = r"H:\Alan Magdaleno Backup\ORPose\camera_poses\chb\camera_extrinsics\aligned_poses.json"

# Expected keys in aligned_poses.json:
# blender2gopro1 ... blender2gopro12
RT_KEY_PATTERN = "blender2gopro{idx}"

# Camera objects will be named:
# GoPro1 ... GoPro12
CAMERA_NAME_PATTERN = "GoPro{idx}"

# Per-camera intrinsics (EDIT this to match your files!)
# Example expected: gopro1_synced_intrinsics.json ... gopro12_synced_intrinsics.json
JSON_K_PATH_PATTERN = r"H:\Alan Magdaleno Backup\ORPose\camera_poses\chb\camera_intrinsics\gopro{idx}intrinsics.json"

# If you want to fall back to ONE intrinsics file when per-camera file is missing, set this:
JSON_K_FALLBACK = None  # e.g. r"...\gopro13_synced_intrinsics.json"
# ===========================================================

# Image resolution used when K was estimated (adjust to your footage)
IMG_W = 3840
IMG_H = 2160

# Sensor size (tweak if you have exact)
SENSOR_WIDTH_MM  = 6.3
SENSOR_HEIGHT_MM = 5.5

# Distortion is ignored (no compositor setup)
USE_COMPOSITOR_DISTORTION = False


# Constant basis change CV(+X,+Y↓,+Z→) → Blender(+X,+Y↑,−Z→)
S_h = Matrix((
    ( 1.0,  0.0,  0.0, 0.0),
    ( 0.0, -1.0,  0.0, 0.0),
    ( 0.0,  0.0, -1.0, 0.0),
    ( 0.0,  0.0,  0.0, 1.0),
))

def load_extrinsics(path: str):
    with open(path, "r") as f:
        return json.load(f)

def load_intrinsics(path: str):
    """Returns fx, fy, cx, cy from the JSON. Ignores distortion completely."""
    with open(path, "r") as f:
        kin = json.load(f)

    # Your structure:
    # kin["sensors"]["RGB"]["intrinsics"]["data"]  # row-major 3x3
    K = kin["sensors"]["RGB"]["intrinsics"]["data"]

    # K = [[fx, 0, cx],
    #      [0, fy, cy],
    #      [0,  0,  1]]
    fx = float(K[0]); cx = float(K[2])
    fy = float(K[4]); cy = float(K[5])
    return fx, fy, cx, cy

def get_or_create_camera(name: str):
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam_data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    return cam

def apply_intrinsics_to_camera(cam_obj, fx, fy, cx, cy):
    cam_obj.data.type = 'PERSP'
    cam_obj.data.sensor_fit = 'HORIZONTAL'
    cam_obj.data.sensor_width  = SENSOR_WIDTH_MM
    cam_obj.data.sensor_height = SENSOR_HEIGHT_MM

    # f_mm = fx * sensor_width / image_width
    f_mm = fx * cam_obj.data.sensor_width / IMG_W
    cam_obj.data.lens = float(f_mm)

    cam_obj.data.shift_x = (cx - (IMG_W * 0.5)) / IMG_W
    cam_obj.data.shift_y = - (cy - (IMG_H * 0.5)) / IMG_H  # flip because CV Y is down

    return f_mm

# Scene render resolution (match K)
scene = bpy.context.scene
scene.render.resolution_x = IMG_W
scene.render.resolution_y = IMG_H
scene.render.pixel_aspect_x = 1
scene.render.pixel_aspect_y = 1

# Extrinsics
extr = load_extrinsics(JSON_RT_PATH)

created = []
missing_rt = []
missing_k  = []

for idx in range(1, 13):
    rt_key = RT_KEY_PATTERN.format(idx=idx)
    cam_name = CAMERA_NAME_PATTERN.format(idx=idx)

    if rt_key not in extr:
        missing_rt.append(rt_key)
        continue

    # world→cam (CV)
    M = Matrix([list(map(float, row)) for row in extr[rt_key]])

    # cam→world (CV) -> Blender basis
    cam_to_world_cv = M.inverted()
    cam_to_world_blender = cam_to_world_cv @ S_h

    cam = get_or_create_camera(cam_name)
    cam.matrix_world = cam_to_world_blender

    # Per-camera intrinsics
    k_path = JSON_K_PATH_PATTERN.format(idx=idx)
    if not os.path.exists(k_path):
        if JSON_K_FALLBACK and os.path.exists(JSON_K_FALLBACK):
            k_path = JSON_K_FALLBACK
        else:
            missing_k.append(k_path)
            print(f"[WARN] Intrinsics file not found for {cam_name}: {k_path} (skipping intrinsics)")
            created.append(cam)
            continue

    fx, fy, cx, cy = load_intrinsics(k_path)
    f_mm = apply_intrinsics_to_camera(cam, fx, fy, cx, cy)

    created.append(cam)
    print(f"[OK] {cam_name}: RT={rt_key}, K={os.path.basename(k_path)}, lens_mm={f_mm:.3f}")

# Select created cameras and set GoPro1 active if present
for o in bpy.context.selected_objects:
    o.select_set(False)
for cam in created:
    cam.select_set(True)

active = bpy.data.objects.get(CAMERA_NAME_PATTERN.format(idx=1)) or (created[0] if created else None)
if active:
    bpy.context.view_layer.objects.active = active

if missing_rt:
    print("[WARN] Missing RT keys:", missing_rt)
if missing_k:
    print("[WARN] Missing intrinsics files:", missing_k)

print(f"Done. Created/updated {len(created)} cameras. (Distortion ignored)")
