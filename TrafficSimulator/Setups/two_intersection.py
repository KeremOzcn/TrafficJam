"""
Two-Intersection Network Setup.

Two horizontally aligned intersections connected by a shared corridor.
Vehicles can travel through both intersections (West→East, East→West)
or enter/exit at the N/S arms of each intersection.

Road index map
--------------
Left intersection  (center at -60, 0):
  0  L_W_IN    external west → left intersection
  1  L_S_IN    external south → left intersection
  2  L_N_IN    external north → left intersection
  3  L_W_OUT   left intersection → external west
  4  L_S_OUT   left intersection → external south
  5  L_N_OUT   left intersection → external north
  6  L_WE_STR  through road going East  (W→E lane inside left intersection)
  7  L_EW_STR  through road going West  (E→W lane inside left intersection)
  8  L_SN_STR  through road going North
  9  L_NS_STR  through road going South

Connector roads (between intersections):
  10 C_EW      going East  (left exit → right entry)
  11 C_WE      going West  (right exit → left entry)

Right intersection (center at +60, 0):
  12 R_E_IN    external east → right intersection
  13 R_S_IN    external south → right intersection
  14 R_N_IN    external north → right intersection
  15 R_E_OUT   right intersection → external east
  16 R_S_OUT   right intersection → external south
  17 R_N_OUT   right intersection → external north
  18 R_WE_STR  through road going East
  19 R_EW_STR  through road going West
  20 R_SN_STR  through road going North
  21 R_NS_STR  through road going South

Signal groups
-------------
Left  signal: group0=[0, 11] (E-W), group1=[1, 2] (N-S)
Right signal: group0=[12, 10] (E-W), group1=[13, 14] (N-S)

Note: connector C_EW (10) is in the RIGHT signal's E-W group so vehicles
on it stop at the right intersection when that signal is red. Likewise
C_WE (11) is in the LEFT signal's E-W group.
"""

from TrafficSimulator import Simulation

a = 2        # lane offset from centreline
b = 12       # inner junction radius
length = 40  # external arm length
cy = 50      # N/S arm length
cx = 60      # intersection centre x-offset

# ── Left intersection: absolute inner nodes ───────────────────────────────
LW_R  = (-cx - b,  a)    # (-72,  2)
LW_L  = (-cx - b, -a)    # (-72, -2)
LE_L  = (-cx + b,  a)    # (-48,  2)
LE_R  = (-cx + b, -a)    # (-48, -2)
LS_R  = (-cx + a,  b)    # (-58, 12)
LS_L  = (-cx - a,  b)    # (-62, 12)
LN_R  = (-cx - a, -b)    # (-62,-12)
LN_L  = (-cx + a, -b)    # (-58,-12)

# Left external endpoints
LW_R_START = (-cx - b - length,  a)    # (-112,  2)
LW_L_START = (-cx - b - length, -a)    # (-112, -2)
LS_R_START = (-cx + a,  b + cy)        # ( -58, 62)
LS_L_START = (-cx - a,  b + cy)        # ( -62, 62)
LN_R_START = (-cx - a, -b - cy)        # ( -62,-62)
LN_L_START = (-cx + a, -b - cy)        # ( -58,-62)

# ── Right intersection: absolute inner nodes ──────────────────────────────
RW_R  = (cx - b,  a)     # ( 48,  2)
RW_L  = (cx - b, -a)     # ( 48, -2)
RE_R  = (cx + b, -a)     # ( 72, -2)
RE_L  = (cx + b,  a)     # ( 72,  2)
RS_R  = (cx + a,  b)     # ( 62, 12)
RS_L  = (cx - a,  b)     # ( 58, 12)
RN_R  = (cx - a, -b)     # ( 58,-12)
RN_L  = (cx + a, -b)     # ( 62,-12)

# Right external endpoints
RE_R_START = (cx + b + length, -a)    # (112, -2)
RE_L_START = (cx + b + length,  a)    # (112,  2)
RS_R_START = (cx + a,  b + cy)        # ( 62, 62)
RS_L_START = (cx - a,  b + cy)        # ( 58, 62)
RN_R_START = (cx - a, -b - cy)        # ( 58,-62)
RN_L_START = (cx + a, -b - cy)        # ( 62,-62)

# ── Road definitions ──────────────────────────────────────────────────────
ROADS = [
    # Left intersection (0–9)
    (LW_R_START, LW_R),    # 0  L_W_IN
    (LS_R_START, LS_R),    # 1  L_S_IN
    (LN_R_START, LN_R),    # 2  L_N_IN
    (LW_L,  LW_L_START),   # 3  L_W_OUT
    (LS_L,  LS_L_START),   # 4  L_S_OUT
    (LN_L,  LN_L_START),   # 5  L_N_OUT
    (LW_R,  LE_L),         # 6  L_WE_STR  (going East)
    (LE_R,  LW_L),         # 7  L_EW_STR  (going West)
    (LS_R,  LN_L),         # 8  L_SN_STR  (going North)
    (LN_R,  LS_L),         # 9  L_NS_STR  (going South)

    # Connectors (10–11)
    (LE_L,  RW_R),         # 10 C_EW  left→right going East  (-48,2)→(48,2)
    (RW_L,  LE_R),         # 11 C_WE  right→left going West  (48,-2)→(-48,-2)

    # Right intersection (12–21)
    (RE_R_START, RE_R),    # 12 R_E_IN
    (RS_R_START, RS_R),    # 13 R_S_IN
    (RN_R_START, RN_R),    # 14 R_N_IN
    (RE_L,  RE_L_START),   # 15 R_E_OUT
    (RS_L,  RS_L_START),   # 16 R_S_OUT
    (RN_L,  RN_L_START),   # 17 R_N_OUT
    (RW_R,  RE_L),         # 18 R_WE_STR  (going East)
    (RE_R,  RW_L),         # 19 R_EW_STR  (going West)
    (RS_R,  RN_L),         # 20 R_SN_STR  (going North)
    (RN_R,  RS_L),         # 21 R_NS_STR  (going South)
]

# ── Vehicle generator ─────────────────────────────────────────────────────
VEHICLE_RATE = 30   # vehicles / minute

PATHS = [
    # weight, [road indices forming the complete route]
    [3, [0,  6, 10, 18, 15]],   # W_ext → E_ext  (through BOTH intersections)
    [3, [12, 19, 11,  7,  3]],  # E_ext → W_ext  (through BOTH intersections)
    [2, [1,  8,  5]],           # S_L → N_L
    [2, [2,  9,  4]],           # N_L → S_L
    [2, [13, 20, 17]],          # S_R → N_R
    [2, [14, 21, 16]],          # N_R → S_R
]
# Total weight = 14  →  7 units enter from west/NS-left, 7 from east/NS-right

# ── Signal configuration ──────────────────────────────────────────────────
# Left signal  — controls roads that queue at the LEFT intersection
L_SIGNAL_ROADS = [[0, 11], [1, 2]]   # group0=E-W, group1=N-S

# Right signal — controls roads that queue at the RIGHT intersection
R_SIGNAL_ROADS = [[12, 10], [13, 14]]  # group0=E-W (incl. connector), group1=N-S

CYCLE        = [(False, True), (False, False), (True, False), (False, False)]
SLOW_DIST    = 50
SLOW_FACTOR  = 0.4
STOP_DIST    = 15

# ── Metadata for agents ───────────────────────────────────────────────────
# Through-roads inside each intersection (used for busy/collision state)
LEFT_THROUGH_ROADS  = {6, 7, 8, 9}
RIGHT_THROUGH_ROADS = {18, 19, 20, 21}

# Connector road indices
CONNECTOR_EW = 10   # going East  (left → right)
CONNECTOR_WE = 11   # going West  (right → left)

# Connector length in simulation units (for propagation delay calculation)
CONNECTOR_LENGTH = 96.0   # LE_L(−48) → RW_R(48)  distance = 96

# Intersection centres (for window offset, optional)
LEFT_CENTRE  = (-cx, 0)
RIGHT_CENTRE = ( cx, 0)

# ── Collision detection ───────────────────────────────────────────────────
INTERSECTIONS_DICT = {
    # Left intersection: E-W straights cross N-S straights
    6:  {8, 9},
    7:  {8, 9},
    8:  {6, 7},
    9:  {6, 7},
    # Right intersection
    18: {20, 21},
    19: {20, 21},
    20: {18, 19},
    21: {18, 19},
}


def two_intersection_setup(max_gen=None) -> Simulation:
    sim = Simulation(max_gen)
    sim.add_roads(ROADS)
    sim.add_generator(VEHICLE_RATE, PATHS)
    sim.add_traffic_signal(L_SIGNAL_ROADS, CYCLE, SLOW_DIST, SLOW_FACTOR, STOP_DIST)
    sim.add_traffic_signal(R_SIGNAL_ROADS, CYCLE, SLOW_DIST, SLOW_FACTOR, STOP_DIST)
    sim.add_intersections(INTERSECTIONS_DICT)
    return sim
