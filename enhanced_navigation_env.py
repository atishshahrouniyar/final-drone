#!/usr/bin/env python3
"""
enhanced_forest_with_obstacles.py

Improved procedural forest scene for PyBullet with added static obstacles.
Run: python enhanced_forest_with_obstacles.py
Press ESC or 'q' in the PyBullet window (or close the window) to exit.

Notes:
 - Tweak SEED, NUM_* constants at the top for density/appearance.
 - ENABLE_SHADOWS can be set True if your GPU renders shadows nicely.
"""

import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import pkg_resources
import os

# For displaying drone camera view
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Drone camera view will be disabled.")

# ---------------- CONFIG ----------------
SEED = 1234
GUI_FPS = 240.0
ARENA_RADIUS = 20.0  # 40x40 playable area (radius 20)
SIMULATION_SPEED = 0.5  # Simulation speed multiplier (0.1 = 10x slower, 1.0 = real-time, 2.0 = 2x faster)

NUM_TREES = 40
NUM_BUSHES = 50
NUM_GRASS = 180
NUM_ROCKS = 24
NUM_LOGS = 8
NUM_MUSHROOM_CLUSTERS = 14
LEAF_CLUMPS = 50
NUM_OBSTACLES = 10  # Number of explicit navigation obstacles

CLEARING = True
CLEARING_RADIUS = 2.0  # keep a small clearing around center (optional)

KEEP_GUI = True   # True -> p.GUI , False -> p.DIRECT
ENABLE_SHADOWS = False  # Toggle shadows for your GPU
# ----------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def rand_in_disk(radius):
    """Return (x,y) uniformly inside disk of given radius."""
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    return r * math.cos(theta), r * math.sin(theta)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EnhancedForestWithObstacles:
    def __init__(self,
                 gui=KEEP_GUI,
                 radius=ARENA_RADIUS,
                 n_trees=NUM_TREES,
                 n_bushes=NUM_BUSHES,
                 n_grass=NUM_GRASS,
                 n_rocks=NUM_ROCKS,
                 n_logs=NUM_LOGS,
                 n_mushrooms=NUM_MUSHROOM_CLUSTERS,
                 leaf_clumps=LEAF_CLUMPS,
                 n_obstacles=NUM_OBSTACLES,
                 clearing=CLEARING,
                 clearing_radius=CLEARING_RADIUS):
        self.radius = radius
        self.n_trees = n_trees
        self.n_bushes = n_bushes
        self.n_grass = n_grass
        self.n_rocks = n_rocks
        self.n_logs = n_logs
        self.n_mushrooms = n_mushrooms
        self.leaf_clumps = leaf_clumps
        self.n_obstacles = n_obstacles
        self.clearing = clearing
        self.clearing_radius = clearing_radius

        # store placed object centers and radius for collision avoidance
        self.placed = []

        # start pybullet
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # GUI tweaks
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if ENABLE_SHADOWS else 0,
                                   physicsClientId=self.client)

        # Build scene
        self._setup_ground()
        self._place_obstacles()         # navigation obstacles (pillars, walls, crates)
        self._place_trees()
        self._place_bushes_and_grass()
        self._place_rocks()
        self._place_logs()
        self._place_mushrooms()
        self._place_leaf_clumps()
        self._place_center_marker()

        # Load drone
        self._load_drone()
        
        # Camera settings for following drone (zoomed in for better visibility)
        self.camera_distance = 4.0  # Distance from drone (reduced for zoom)
        self.camera_pitch = -30.0   # Camera angle (less steep for better view)
        self.camera_yaw_offset = 45.0  # Camera yaw offset (degrees)
        
        # Drone scale factor (to make drone bigger/more visible)
        self.drone_scale = 2.0  # Scale factor for drone size
        
        # Drone camera settings
        self.show_drone_camera = True  # Enable/disable drone camera view
        self.drone_camera_width = 480  # Camera image width
        self.drone_camera_height = 360  # Camera image height
        self.drone_camera_fov = 80  # Field of view (degrees)
        self.drone_camera_update_freq = 2  # Update camera every N frames (for performance)
        self.drone_camera_frame_count = 0
        
        # Display mode: 'window' (separate OpenCV window), 'split' (side-by-side in PyBullet), or 'both'
        self.camera_display_mode = 'split'  # 'window', 'split', or 'both'
        
        # For split-screen rendering in PyBullet
        self.use_split_screen = True  # Render drone view in PyBullet window
        
        # Simulation speed control
        self.simulation_speed = SIMULATION_SPEED  # Speed multiplier (0.1-2.0)
        self.initial_fps = GUI_FPS  # Store original FPS for reference
        
        # Initial camera setup
        self._update_camera()

    def _valid(self, x, y, min_sep=0.8):
        """Simple rejection check to avoid heavy overlaps; keeps clearing if requested."""
        if self.clearing and math.hypot(x, y) < self.clearing_radius:
            return False
        for ox, oy, r in self.placed:
            if dist((x, y), (ox, oy)) < (r + min_sep):
                return False
        return True

    def _setup_ground(self):
        """Single plane as ground. Try to apply grass texture, otherwise color fallback."""
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        try:
            tex = p.loadTexture("grass.png", physicsClientId=self.client)  # pybullet_data contains grass.png
            p.changeVisualShape(plane_id, -1, textureUniqueId=tex, physicsClientId=self.client)
        except Exception:
            # fallback green color
            p.changeVisualShape(plane_id, -1, rgbaColor=[0.16, 0.56, 0.18, 1.0], physicsClientId=self.client)

    def _place_obstacles(self):
        """
        Place a few static obstacles to act as navigation challenges:
         - tall pillars (cylinders)
         - low walls (thin boxes)
         - crates / boxes
        """
        for i in range(self.n_obstacles):
            for _ in range(40):
                x, y = rand_in_disk(self.radius * 0.93)
                if not self._valid(x, y, min_sep=1.8): 
                    continue
                # cycle obstacle types for variety
                typ = i % 3
                if typ == 0:
                    # Pillar (cylinder)
                    height = random.uniform(1.6, 3.8)
                    radius = random.uniform(0.22, 0.48)
                    color = random.choice([[0.86, 0.17, 0.17, 1], [0.17, 0.3, 0.86, 1], [0.95, 0.55, 0.16, 1]])
                    coll = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                                  physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color,
                                              physicsClientId=self.client)
                    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                      basePosition=[x, y, height/2.0], physicsClientId=self.client)
                    self.placed.append((x, y, max(radius, 0.6)))
                elif typ == 1:
                    # Low wall (thin box) — can act like a curb or broken wall
                    length = random.uniform(1.0, 3.0)
                    thickness = random.uniform(0.12, 0.30)
                    height = random.uniform(0.6, 1.6)
                    yaw = random.uniform(0, math.pi)
                    color = random.choice([[0.9, 0.88, 0.26, 1], [0.68, 0.5, 0.4, 1]])
                    coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, height/2],
                                                  physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, height/2],
                                              rgbaColor=color, physicsClientId=self.client)
                    orn = p.getQuaternionFromEuler([0, 0, yaw])
                    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                      basePosition=[x, y, height/2.0], baseOrientation=orn,
                                      physicsClientId=self.client)
                    self.placed.append((x, y, max(length/2, 0.8)))
                else:
                    # Crate / box
                    side = random.uniform(0.4, 1.2)
                    height = random.uniform(0.4, 1.0)
                    color = random.choice([[0.56, 0.25, 0.75, 1], [0.36, 0.26, 0.22, 1]])
                    coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[side/2, side/2, height/2],
                                                  physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[side/2, side/2, height/2],
                                              rgbaColor=color, physicsClientId=self.client)
                    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                      basePosition=[x, y, height/2.0], physicsClientId=self.client)
                    self.placed.append((x, y, side/1.8))
                break

    def _place_trees(self):
        """Place trees (trunk + stacked sphere crowns)."""
        attempts = 0
        placed_count = 0
        max_attempts = self.n_trees * 50
        while placed_count < self.n_trees and attempts < max_attempts:
            attempts += 1
            x, y = rand_in_disk(self.radius * 0.9)
            if not self._valid(x, y, min_sep=0.9):
                continue
            trunk_h = random.uniform(1.6, 3.2)
            trunk_r = random.uniform(0.12, 0.26)
            crown = random.uniform(0.7, 1.6)

            trunk_col = [0.28 + random.uniform(-0.05, 0.05),
                         0.16 + random.uniform(-0.04, 0.04),
                         0.07 + random.uniform(-0.02, 0.02), 1.0]
            crown_col = [0.07 + random.uniform(0.0, 0.12),
                         0.40 + random.uniform(-0.05, 0.18),
                         0.07 + random.uniform(-0.02, 0.06), 1.0]

            # add tree and mark radius for placement avoidance
            self._add_tree(x, y, trunk_h, trunk_r, crown, trunk_col, crown_col)
            body_radius = crown * 1.05
            self.placed.append((x, y, body_radius))
            placed_count += 1

    def _add_tree(self, x, y, trunk_h, trunk_r, crown_size, trunk_col, crown_col):
        """Create tree trunk and crown (collision + visual shapes)."""
        trunk_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=trunk_r, height=trunk_h,
                                                 physicsClientId=self.client)
        trunk_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=trunk_r, length=trunk_h,
                                           rgbaColor=trunk_col, physicsClientId=self.client)
        trunk_body = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=trunk_collision,
                                       baseVisualShapeIndex=trunk_visual,
                                       basePosition=[x, y, trunk_h / 2.0],
                                       physicsClientId=self.client)
        # reduce specular
        try:
            p.changeVisualShape(trunk_body, -1, specularColor=[0.06, 0.06, 0.06], physicsClientId=self.client)
        except Exception:
            pass

        # stacked spheres as crown with slight offset jitter to avoid perfect coplanar surfaces
        n_spheres = random.randint(2, 4)
        for i in range(n_spheres):
            offset_x = np.random.normal(scale=0.12)
            offset_y = np.random.normal(scale=0.12)
            scale = crown_size * random.uniform(0.65, 1.05) * (1.0 - i * 0.07)
            z = trunk_h + i * (scale * 0.55) + 0.02 + random.uniform(-0.02, 0.02)
            coll = p.createCollisionShape(p.GEOM_SPHERE, radius=scale, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=scale,
                                      rgbaColor=[crown_col[0] * (0.9 + random.random() * 0.15),
                                                 min(1.0, crown_col[1] * (0.9 + random.random() * 0.15)),
                                                 crown_col[2] * (0.9 + random.random() * 0.15), 1.0],
                                      physicsClientId=self.client)
            crown_body = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=coll,
                                           baseVisualShapeIndex=vis,
                                           basePosition=[x + offset_x, y + offset_y, z],
                                           physicsClientId=self.client)
            try:
                p.changeVisualShape(crown_body, -1, specularColor=[0.06, 0.06, 0.06], physicsClientId=self.client)
            except Exception:
                pass

    def _place_bushes_and_grass(self):
        """Add bushes and small grass/foliage."""
        placed = 0
        attempts = 0
        while placed < self.n_bushes and attempts < self.n_bushes * 20:
            attempts += 1
            x, y = rand_in_disk(self.radius * 0.95)
            if not self._valid(x, y, min_sep=0.35):
                continue
            size = random.uniform(0.18, 0.46)
            color = [0.08 + random.random() * 0.18,
                     0.42 + random.random() * 0.28,
                     0.08 + random.random() * 0.12, 1.0]
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color, physicsClientId=self.client)
            body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[x, y, size * 0.45],
                                     physicsClientId=self.client)
            try:
                p.changeVisualShape(body, -1, specularColor=[0.05, 0.05, 0.05], physicsClientId=self.client)
            except Exception:
                pass
            self.placed.append((x, y, size))
            placed += 1

        # grass blades / small tufts
        grass_placed = 0
        for _ in range(self.n_grass):
            x, y = rand_in_disk(self.radius * 0.98)
            if not self._valid(x, y, min_sep=0.03):
                continue
            h = random.uniform(0.05, 0.18)
            w = random.uniform(0.02, 0.04)
            color = [0.08 + random.random() * 0.16, 0.56 + random.random() * 0.2, 0.08 + random.random() * 0.1, 1.0]
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w, w * 1.2, h], rgbaColor=color,
                                      physicsClientId=self.client)
            yaw = random.random() * 2 * math.pi
            orn = p.getQuaternionFromEuler([random.uniform(-0.12, 0.12), 0.0, yaw])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[x, y, h], baseOrientation=orn,
                              physicsClientId=self.client)
            grass_placed += 1
        # (grass not added to placed[] to keep density high)

    def _place_rocks(self):
        """Place rocks/boulders with slight shape and color variety."""
        for _ in range(self.n_rocks):
            attempts = 0
            while attempts < 6:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.9)
                if not self._valid(x, y, min_sep=0.15):
                    continue
                if random.random() < 0.6:
                    r = random.uniform(0.12, 0.45)
                    sx = random.uniform(0.7, 1.3)
                    sy = random.uniform(0.7, 1.3)
                    sz = random.uniform(0.6, 1.0)
                    coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[r * sx, r * sy, r * sz],
                                                  physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                              rgbaColor=[0.45 + random.random() * 0.18,
                                                         0.45 + random.random() * 0.18,
                                                         0.42 + random.random() * 0.12, 1],
                                              physicsClientId=self.client)
                    yaw = random.random() * 2 * math.pi
                    orn = p.getQuaternionFromEuler([0, 0, yaw])
                    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll,
                                             baseVisualShapeIndex=vis,
                                             basePosition=[x, y, r * 0.8], baseOrientation=orn,
                                             physicsClientId=self.client)
                    try:
                        p.changeVisualShape(body, -1, specularColor=[0.04, 0.04, 0.04], physicsClientId=self.client)
                    except Exception:
                        pass
                    self.placed.append((x, y, r * max(sx, sy) * 1.05))
                    break
                else:
                    hx = random.uniform(0.08, 0.6)
                    hy = random.uniform(0.08, 0.6)
                    hz = random.uniform(0.05, 0.32)
                    coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                              rgbaColor=[0.4 + random.random() * 0.24,
                                                         0.4 + random.random() * 0.18,
                                                         0.38 + random.random() * 0.16, 1],
                                              physicsClientId=self.client)
                    yaw = random.random() * 2 * math.pi
                    orn = p.getQuaternionFromEuler([random.uniform(-0.18, 0.18), random.uniform(-0.12, 0.12), yaw])
                    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                             basePosition=[x, y, hz], baseOrientation=orn, physicsClientId=self.client)
                    try:
                        p.changeVisualShape(body, -1, specularColor=[0.04, 0.04, 0.04], physicsClientId=self.client)
                    except Exception:
                        pass
                    self.placed.append((x, y, max(hx, hy) * 1.05))
                    break

    def _place_logs(self):
        """Place fallen logs (cylinders oriented roughly horizontally)."""
        for _ in range(self.n_logs):
            attempts = 0
            while attempts < 8:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.9)
                if not self._valid(x, y, min_sep=0.9):
                    continue
                length = random.uniform(0.8, 2.0)
                radius = random.uniform(0.075, 0.17)
                coll = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length,
                                              physicsClientId=self.client)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length,
                                          rgbaColor=[0.33 + random.random() * 0.06, 0.20 + random.random() * 0.04,
                                                     0.10 + random.random() * 0.03, 1], physicsClientId=self.client)
                yaw = random.random() * 2 * math.pi
                pitch = random.uniform(-0.35, 0.35)
                orn = p.getQuaternionFromEuler([pitch, 0, yaw])
                body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll,
                                         baseVisualShapeIndex=vis,
                                         basePosition=[x, y, radius * 0.45], baseOrientation=orn,
                                         physicsClientId=self.client)
                try:
                    p.changeVisualShape(body, -1, specularColor=[0.04, 0.04, 0.04], physicsClientId=self.client)
                except Exception:
                    pass
                self.placed.append((x, y, length * 0.6))
                break

    def _place_mushrooms(self):
        """Place small mushroom clusters on the forest floor."""
        for _ in range(self.n_mushrooms):
            attempts = 0
            while attempts < 6:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.95)
                if not self._valid(x, y, min_sep=0.15):
                    continue
                cluster_size = random.randint(2, 8)
                base_x = x
                base_y = y
                for i in range(cluster_size):
                    dx = np.random.normal(scale=0.10)
                    dy = np.random.normal(scale=0.10)
                    stem_h = random.uniform(0.05, 0.12)
                    stem_r = random.uniform(0.01, 0.03)
                    cap_r = random.uniform(0.03, 0.07)
                    stem_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=stem_r, length=stem_h,
                                                   rgbaColor=[0.95, 0.9, 0.82, 1], physicsClientId=self.client)
                    p.createMultiBody(baseMass=0, baseVisualShapeIndex=stem_vis,
                                      basePosition=[base_x + dx, base_y + dy, stem_h / 2.0],
                                      physicsClientId=self.client)
                    cap_vis = p.createVisualShape(p.GEOM_SPHERE, radius=cap_r,
                                                  rgbaColor=[0.9, 0.2 + random.random() * 0.3, 0.18 + random.random() * 0.18, 1],
                                                  physicsClientId=self.client)
                    p.createMultiBody(baseMass=0, baseVisualShapeIndex=cap_vis,
                                      basePosition=[base_x + dx, base_y + dy, stem_h + cap_r * 0.28],
                                      physicsClientId=self.client)
                self.placed.append((x, y, 0.2))
                break

    def _place_leaf_clumps(self):
        """Scatter low-height leaf clump geometry on the ground to enrich detail."""
        for _ in range(self.leaf_clumps):
            attempts = 0
            while attempts < 6:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.95)
                if not self._valid(x, y, min_sep=0.07):
                    continue
                count = random.randint(6, 20)
                for i in range(count):
                    dx = np.random.normal(scale=0.18)
                    dy = np.random.normal(scale=0.18)
                    w = random.uniform(0.02, 0.06)
                    h = 0.003 + random.uniform(0.0, 0.01)
                    color = [0.25 + random.random() * 0.35, 0.35 + random.random() * 0.35, 0.06 + random.random() * 0.2, 1]
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w, w * 1.2, h], rgbaColor=color,
                                              physicsClientId=self.client)
                    yaw = random.random() * 2 * math.pi
                    orn = p.getQuaternionFromEuler([0, 0, yaw])
                    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                      basePosition=[x + dx, y + dy, h], baseOrientation=orn, physicsClientId=self.client)
                self.placed.append((x, y, 0.25))
                break

    def _place_center_marker(self):
        """Add small start marker and text so you can see spawn center."""
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0.08, 0.7, 0.08, 0.95],
                                  physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0.08], physicsClientId=self.client)
        try:
            p.addUserDebugText("START", [0.0, -0.8, 0.02], textColorRGB=[0.02, 0.6, 0.02], textSize=1.2,
                               physicsClientId=self.client)
        except Exception:
            pass

    def _load_drone(self):
        """Load a drone at the center of the clearing."""
        # Try to find the drone URDF file
        drone_urdf_paths = []
        
        # Try gym_pybullet_drones package (with error handling)
        try:
            gym_drones_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/cf2x.urdf')
            drone_urdf_paths.append(gym_drones_path)
        except (ImportError, ModuleNotFoundError, Exception):
            # Package not installed, skip this path
            pass
        
        # Try relative paths to common locations
        drone_urdf_paths.extend([
            '../gym-pybullet-drones 2/gym_pybullet_drones/assets/cf2x.urdf',
            '../testing-Agents/gym_pybullet_drones/assets/cf2x.urdf',
            '../../testing-Agents/gym_pybullet_drones/assets/cf2x.urdf',
            # Try absolute path
            os.path.join(os.path.dirname(__file__), '../testing-Agents/gym_pybullet_drones/assets/cf2x.urdf'),
        ])
        
        drone_urdf = None
        for path in drone_urdf_paths:
            if path and os.path.exists(path):
                drone_urdf = path
                break
        
        scale = getattr(self, 'drone_scale', 2.0)  # Get scale factor
        
        if drone_urdf is None:
            print("Warning: Could not find cf2x.urdf. Creating a simple drone placeholder.")
            # Create a simple drone placeholder (box with 4 propellers)
            self.drone_id = self._create_simple_drone([0, 0, 1.0])
        else:
            # Load the actual drone URDF
            print(f"Loading drone from: {drone_urdf}")
            init_position = [0, 0, 1.0]  # Start at center, 1m above ground
            init_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            self.drone_id = p.loadURDF(
                drone_urdf,
                init_position,
                init_orientation,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.client
            )
            
            # Add a visible marker sphere around the drone to make it more prominent
            # PyBullet doesn't easily scale URDF models, so we use a visual aid
            if scale != 1.0:
                print(f"Adding visual marker for drone visibility (scale factor: {scale})")
                # Create a semi-transparent sphere marker that will follow the drone
                # This helps visualize the drone's position even when the URDF model is small
                self.drone_marker_vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.15 * scale,  # Visible marker radius
                    rgbaColor=[1.0, 0.3, 0.3, 0.3],  # Semi-transparent red marker
                    physicsClientId=self.client
                )
                # The marker will be updated in the run loop to follow the drone
                self.drone_marker_id = None  # Will be created/updated in _update_camera or run loop
        
        # Store initial position for reference
        self.drone_position = np.array([0, 0, 1.0])
        self.drone_velocity = np.array([0, 0, 0])
        
        # Drone control parameters (simple hover control)
        self.hover_height = 1.5
        self.target_position = np.array([0, 0, self.hover_height])
        self._start_time = None  # Initialize start time for movement pattern

    def _create_simple_drone(self, position):
        """Create a simple drone placeholder if URDF is not available."""
        scale = getattr(self, 'drone_scale', 2.0)  # Use scale factor if available
        
        # Main body (scaled up box)
        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05 * scale, 0.05 * scale, 0.02 * scale],
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
            physicsClientId=self.client
        )
        body_coll = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.05 * scale, 0.05 * scale, 0.02 * scale],
            physicsClientId=self.client
        )
        
        body_id = p.createMultiBody(
            baseMass=0.027 * (scale ** 3),  # Mass scales with volume
            baseCollisionShapeIndex=body_coll,
            baseVisualShapeIndex=body_vis,
            basePosition=position,
            physicsClientId=self.client
        )
        
        # Add 4 propellers (scaled up spheres) at corners
        prop_positions = [
            [0.04 * scale, 0.04 * scale, 0.03 * scale],
            [-0.04 * scale, 0.04 * scale, 0.03 * scale],
            [0.04 * scale, -0.04 * scale, 0.03 * scale],
            [-0.04 * scale, -0.04 * scale, 0.03 * scale]
        ]
        prop_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02 * scale,
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
            physicsClientId=self.client
        )
        
        for prop_pos in prop_positions:
            prop_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=prop_vis,
                basePosition=[position[0] + prop_pos[0], position[1] + prop_pos[1], position[2] + prop_pos[2]],
                physicsClientId=self.client
            )
        
        return body_id

    def _update_camera(self):
        """Update camera to follow the drone position dynamically."""
        if not hasattr(self, 'drone_id'):
            return
        
        try:
            # Get current drone position and velocity
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
            
            self.drone_position = np.array(pos)
            self.drone_velocity = np.array(vel)
            
            # Update visible marker if it exists (for URDF drones to make them more visible)
            if hasattr(self, 'drone_marker_vis') and self.drone_marker_vis is not None:
                try:
                    if hasattr(self, 'drone_marker_id') and self.drone_marker_id is not None:
                        # Update existing marker position
                        p.resetBasePositionAndOrientation(
                            self.drone_marker_id,
                            pos,  # Same position as drone
                            [0, 0, 0, 1],  # No rotation
                            physicsClientId=self.client
                        )
                    else:
                        # Create marker for first time
                        scale = getattr(self, 'drone_scale', 2.0)
                        self.drone_marker_id = p.createMultiBody(
                            baseMass=0,
                            baseVisualShapeIndex=self.drone_marker_vis,
                            basePosition=pos,
                            physicsClientId=self.client
                        )
                except Exception:
                    pass
            
            # Camera always targets the drone's current position
            camera_target = pos
            
            # Calculate camera yaw based on drone's movement direction
            # If drone is moving, follow its direction; otherwise use default offset
            velocity_magnitude = np.linalg.norm(self.drone_velocity)
            if velocity_magnitude > 0.1:
                # Camera follows behind the drone in the direction it's moving
                movement_direction = math.atan2(self.drone_velocity[1], self.drone_velocity[0])
                camera_yaw = math.degrees(movement_direction) + 180.0  # Behind the drone
            else:
                # If stationary, use default view angle
                camera_yaw = self.camera_yaw_offset
            
            # Smoothly update camera to follow drone
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=camera_target,
                physicsClientId=self.client
            )
        except Exception as e:
            # If drone doesn't exist yet, use default camera
            try:
                p.resetDebugVisualizerCamera(
                    cameraDistance=12,
                    cameraYaw=45,
                    cameraPitch=-35,
                    cameraTargetPosition=[0, 0, 0],
                    physicsClientId=self.client
                )
            except Exception:
                pass

    def _control_drone(self):
        """Simple control to make drone hover and move around."""
        try:
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
            
            self.drone_position = np.array(pos)
            self.drone_velocity = np.array(vel)
            
            # Simple PID-like control for hover
            height_error = self.hover_height - pos[2]
            desired_force_z = height_error * 5.0 - vel[2] * 2.0  # P-D control
            
            # Add some horizontal movement (circular pattern)
            time_step = p.getPhysicsEngineParameters(physicsClientId=self.client)['fixedTimeStep']
            # Initialize start time if not already set
            if self._start_time is None:
                self._start_time = time.time()
                t = 0.0
            else:
                t = time.time() - self._start_time
            
            # Circular motion pattern
            radius = 3.0
            angular_vel = 0.3
            target_x = radius * math.cos(angular_vel * t)
            target_y = radius * math.sin(angular_vel * t)
            target_z = self.hover_height
            
            # Move toward target
            pos_error = np.array([target_x, target_y, target_z]) - self.drone_position
            desired_force = pos_error * 2.0 - self.drone_velocity * 1.0
            
            # Apply forces (simplified - in real drone, this would control rotors)
            # For now, we'll apply forces directly to make it move
            max_force = 10.0
            force = np.clip(desired_force, -max_force, max_force)
            
            # Apply force to drone body
            p.applyExternalForce(
                self.drone_id,
                -1,  # -1 means apply to base
                force.tolist(),
                [0, 0, 0],  # position (relative to COM, 0 means at COM)
                p.WORLD_FRAME,
                physicsClientId=self.client
            )
            
            # Add some upward force to counteract gravity (scaled by mass)
            drone_mass = p.getDynamicsInfo(self.drone_id, -1, physicsClientId=self.client)[0]
            if drone_mass > 0:
                hover_force = 9.81 * drone_mass
            else:
                # Fallback to estimated mass
                scale = getattr(self, 'drone_scale', 2.0)
                hover_force = 9.81 * 0.027 * (scale ** 3)
            
            p.applyExternalForce(
                self.drone_id,
                -1,
                [0, 0, hover_force],
                [0, 0, 0],
                p.WORLD_FRAME,
                physicsClientId=self.client
            )
            
        except Exception as e:
            print(f"Error controlling drone: {e}")

    def _get_drone_camera_image(self):
        """Capture image from drone's onboard camera (first-person view)."""
        if not hasattr(self, 'drone_id') or not self.show_drone_camera:
            return None
        
        try:
            # Get drone position and orientation
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            
            # Convert quaternion to rotation matrix to get forward direction
            quat = orn
            rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            
            # Camera position (slightly forward and up from drone center)
            # Offset forward by drone size, up a bit to avoid seeing the drone body
            camera_offset = np.array([0.1, 0, 0.05])  # Forward (x), right (y), up (z)
            camera_pos = pos + np.dot(rot_matrix, camera_offset)
            
            # Target point (looking forward in drone's direction)
            forward_vec = np.dot(rot_matrix, np.array([1, 0, 0]))  # Forward direction
            target_pos = camera_pos + forward_vec * 5.0  # Look 5m ahead
            
            # Up vector (should be drone's up direction)
            up_vec = np.dot(rot_matrix, np.array([0, 0, 1]))  # Up direction
            
            # Compute view matrix
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos.tolist(),
                cameraTargetPosition=target_pos.tolist(),
                cameraUpVector=up_vec.tolist(),
                physicsClientId=self.client
            )
            
            # Compute projection matrix
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.drone_camera_fov,
                aspect=self.drone_camera_width / self.drone_camera_height,
                nearVal=0.01,
                farVal=100.0,
                physicsClientId=self.client
            )
            
            # Get camera image (width, height, RGB, depth, segmentation)
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=self.drone_camera_width,
                height=self.drone_camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.client
            )
            
            # Convert RGB image from uint8 to proper format
            # PyBullet returns RGBA, we need to convert to RGB and flip vertically
            rgb_array = np.array(rgb_img, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel (keep only RGB)
            rgb_array = np.flipud(rgb_array)  # Flip vertically (PyBullet's image is upside down)
            
            return rgb_array
            
        except Exception as e:
            print(f"Error capturing drone camera: {e}")
            return None

    def _display_drone_camera_opencv(self, img):
        """Display drone camera image in a separate OpenCV window."""
        if img is None or not CV2_AVAILABLE:
            return
        
        try:
            # Resize for display if needed (for better visibility)
            display_img = cv2.resize(img, (640, 480))
            
            # Add text overlay
            cv2.putText(display_img, "DRONE CAMERA VIEW", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, "Press 'c' to toggle camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display image
            cv2.imshow("Drone Camera", display_img)
            cv2.waitKey(1)  # Required for OpenCV to update the window
            
        except Exception as e:
            print(f"Error displaying drone camera in OpenCV: {e}")

    def _render_drone_camera_split_screen(self):
        """Render drone camera view in split-screen mode within PyBullet window."""
        if not hasattr(self, 'drone_id') or not self.show_drone_camera:
            return
        
        try:
            # Get drone position and orientation
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            
            # Convert quaternion to rotation matrix
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            
            # Camera position (forward and up from drone)
            camera_offset = np.array([0.1, 0, 0.05])
            camera_pos = pos + np.dot(rot_matrix, camera_offset)
            
            # Target point (forward direction)
            forward_vec = np.dot(rot_matrix, np.array([1, 0, 0]))
            target_pos = camera_pos + forward_vec * 5.0
            
            # Up vector
            up_vec = np.dot(rot_matrix, np.array([0, 0, 1]))
            
            # Get window dimensions for split screen
            # PyBullet window is typically rendered at specific size
            # We'll render the drone view in a small overlay in the corner
            viewport_width = 1024  # Approximate PyBullet window width
            viewport_height = 768  # Approximate PyBullet window height
            
            # Render drone view in top-right corner (small overlay)
            overlay_width = int(viewport_width * 0.3)  # 30% of window width
            overlay_height = int(viewport_height * 0.3)  # 30% of window height
            
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos.tolist(),
                cameraTargetPosition=target_pos.tolist(),
                cameraUpVector=up_vec.tolist(),
                physicsClientId=self.client
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.drone_camera_fov,
                aspect=float(overlay_width) / overlay_height,
                nearVal=0.01,
                farVal=100.0,
                physicsClientId=self.client
            )
            
            # Render to a render target and then copy to debug visualizer
            # Note: PyBullet doesn't directly support overlay rendering, so we'll use OpenCV
            # But we can at least capture the image for display
            
            # Alternative: Use PyBullet's debug visualizer with custom rendering
            # We'll use the image data we already capture and display it via OpenCV overlay
            
        except Exception as e:
            print(f"Error in split-screen rendering: {e}")

    def _display_drone_camera(self, img):
        """Display drone camera image based on display mode."""
        if img is None:
            return
        
        # Display in OpenCV window if enabled
        if self.camera_display_mode in ['window', 'both'] and CV2_AVAILABLE:
            self._display_drone_camera_opencv(img)
        
        # For split screen, we'll use OpenCV to create a side-by-side view
        # PyBullet doesn't natively support split-screen overlays, so we create a composite window
        if self.camera_display_mode in ['split', 'both'] and CV2_AVAILABLE:
            try:
                # Create a larger window showing both views side-by-side
                # Left side: Main PyBullet view (we'll note this in the composite)
                # Right side: Drone camera view
                
                # Resize drone image for split view
                drone_img_resized = cv2.resize(img, (640, 480))
                
                # Add border and text to drone view
                cv2.rectangle(drone_img_resized, (0, 0), (drone_img_resized.shape[1]-1, drone_img_resized.shape[0]-1), (0, 255, 0), 3)
                cv2.putText(drone_img_resized, "DRONE CAMERA VIEW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(drone_img_resized, "Press 'c' to toggle", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Create a placeholder for main view (black with text)
                main_view = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(main_view, "PYBULLET MAIN VIEW", (150, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(main_view, "(Check PyBullet window)", (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Combine side-by-side
                split_view = np.hstack([main_view, drone_img_resized])
                
                # Display split view
                cv2.imshow("Split View - Main | Drone Camera", split_view)
                cv2.waitKey(1)
                
            except Exception as e:
                print(f"Error displaying split-screen view: {e}")

    def run(self, fps=GUI_FPS):
        """Main loop — keeps GUI open until ESC or 'q' pressed or window closed."""
        base_dt = 1.0 / fps  # Base timestep
        dt = base_dt / self.simulation_speed  # Adjusted timestep based on speed
        
        print("Scene running. Press ESC or 'q' in the PyBullet window or close the window to exit.")
        print("Drone is flying with a following camera view.")
        print(f"Simulation speed: {self.simulation_speed}x ({'SLOW' if self.simulation_speed < 0.5 else 'NORMAL' if self.simulation_speed < 1.5 else 'FAST'})")
        print("Controls:")
        print("  Press 'c' to toggle drone camera")
        print("  Press 'm' to change display mode")
        print("  Press '+' or '=' to speed up simulation")
        print("  Press '-' or '_' to slow down simulation")
        print("  Press '0' to reset to normal speed")
        
        if CV2_AVAILABLE and self.show_drone_camera:
            print("Drone camera view enabled. Press 'c' in PyBullet window to toggle.")
            print(f"Display mode: {self.camera_display_mode}")
            if self.camera_display_mode == 'split':
                print("Split-screen view: Check OpenCV window for side-by-side display")
            elif self.camera_display_mode == 'window':
                print("Separate window: Check 'Drone Camera' OpenCV window")
            elif self.camera_display_mode == 'both':
                print("Both modes: Check both OpenCV windows")
        elif not CV2_AVAILABLE:
            print("Warning: OpenCV not available. Install with: pip install opencv-python")
        try:
            while True:
                if not p.isConnected(physicsClientId=self.client):
                    break
                keys = p.getKeyboardEvents()

                # ESC key or 'q' to quit
                if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                    print("ESC pressed. Exiting.")
                    break
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    print("'q' pressed. Exiting.")
                    break
                
                # Toggle drone camera with 'c' key
                if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                    self.show_drone_camera = not self.show_drone_camera
                    if self.show_drone_camera:
                        print("Drone camera view: ENABLED")
                        print(f"  Display mode: {self.camera_display_mode}")
                    else:
                        print("Drone camera view: DISABLED")
                        if CV2_AVAILABLE:
                            try:
                                cv2.destroyWindow("Drone Camera")
                                cv2.destroyWindow("Split View - Main | Drone Camera")
                            except:
                                pass
                
                # Toggle display mode with 'm' key
                if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED and CV2_AVAILABLE:
                    modes = ['split', 'window', 'both']
                    current_idx = modes.index(self.camera_display_mode) if self.camera_display_mode in modes else 0
                    next_idx = (current_idx + 1) % len(modes)
                    self.camera_display_mode = modes[next_idx]
                    print(f"Display mode changed to: {self.camera_display_mode}")
                
                # Speed control with '+' and '-' keys
                if ord('+') in keys and keys[ord('+')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = min(2.0, self.simulation_speed + 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('=') in keys and keys[ord('=')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = min(2.0, self.simulation_speed + 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('-') in keys and keys[ord('-')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = max(0.1, self.simulation_speed - 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('_') in keys and keys[ord('_')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = max(0.1, self.simulation_speed - 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = 1.0
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed reset to: {self.simulation_speed:.1f}x (normal)")

                # Control drone movement
                if hasattr(self, 'drone_id'):
                    self._control_drone()
                
                # Update camera to follow drone
                self._update_camera()
                
                # Capture and display drone camera view
                if self.show_drone_camera and CV2_AVAILABLE:
                    self.drone_camera_frame_count += 1
                    if self.drone_camera_frame_count >= self.drone_camera_update_freq:
                        drone_img = self._get_drone_camera_image()
                        if drone_img is not None:
                            self._display_drone_camera(drone_img)
                        self.drone_camera_frame_count = 0

                # step simulation and wait (adjusted for speed)
                p.stepSimulation(physicsClientId=self.client)
                time.sleep(dt)
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            # Clean up OpenCV windows
            if CV2_AVAILABLE:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            try:
                p.disconnect(physicsClientId=self.client)
            except Exception:
                pass
            print("Disconnected PyBullet.")


if __name__ == "__main__":
    scene = EnhancedForestWithObstacles(
        gui=KEEP_GUI,
        radius=ARENA_RADIUS,
        n_trees=NUM_TREES,
        n_bushes=NUM_BUSHES,
        n_grass=NUM_GRASS,
        n_rocks=NUM_ROCKS,
        n_logs=NUM_LOGS,
        n_mushrooms=NUM_MUSHROOM_CLUSTERS,
        leaf_clumps=LEAF_CLUMPS,
        n_obstacles=NUM_OBSTACLES,
        clearing=CLEARING,
        clearing_radius=CLEARING_RADIUS
    )
    scene.run(fps=GUI_FPS)
