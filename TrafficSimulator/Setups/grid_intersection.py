"""
2x2 Grid Intersection Setup.

Four intersections arranged in a square grid:
    [TL] --- [TR]
     |          |
    [BL] --- [BR]

External arms only on the outer edges of the grid:
  TL: West and North
  TR: East  and North
  BL: West  and South
  BR: East  and South

Adjacent intersections are connected by bidirectional corridor roads.

Road index map
--------------
TL intersection (centre -cx, -cy = -60, -60):
   0  TL_W_IN    external west  -> TL
   1  TL_N_IN    external north -> TL
   2  TL_W_OUT   TL -> external west
   3  TL_N_OUT   TL -> external north
   4  TL_WE_STR  through road going East
   5  TL_EW_STR  through road going West
   6  TL_SN_STR  through road going North
   7  TL_NS_STR  through road going South

TR intersection (centre +cx, -cy = +60, -60):
   8  TR_E_IN    external east  -> TR (arriving going west)
   9  TR_N_IN    external north -> TR
  10  TR_E_OUT   TR -> external east
  11  TR_N_OUT   TR -> external north
  12  TR_WE_STR  through road going East
  13  TR_EW_STR  through road going West
  14  TR_SN_STR  through road going North
  15  TR_NS_STR  through road going South

BL intersection (centre -cx, +cy = -60, +60):
  16  BL_W_IN    external west  -> BL
  17  BL_S_IN    external south -> BL
  18  BL_W_OUT   BL -> external west
  19  BL_S_OUT   BL -> external south
  20  BL_WE_STR  through road going East
  21  BL_EW_STR  through road going West
  22  BL_SN_STR  through road going North
  23  BL_NS_STR  through road going South

BR intersection (centre +cx, +cy = +60, +60):
  24  BR_E_IN    external east  -> BR (arriving going west)
  25  BR_S_IN    external south -> BR
  26  BR_E_OUT   BR -> external east
  27  BR_S_OUT   BR -> external south
  28  BR_WE_STR  through road going East
  29  BR_EW_STR  through road going West
  30  BR_SN_STR  through road going North
  31  BR_NS_STR  through road going South

Connector roads (32-39):
  32  C_TL_TR   TL east exit -> TR west entry   (going East,  top    horizontal)
  33  C_TR_TL   TR west exit -> TL east entry   (going West,  top    horizontal)
  34  C_BL_BR   BL east exit -> BR west entry   (going East,  bottom horizontal)
  35  C_BR_BL   BR west exit -> BL east entry   (going West,  bottom horizontal)
  36  C_TL_BL   TL south exit -> BL north entry (going South, left   vertical)
  37  C_BL_TL   BL north exit -> TL south entry (going North, left   vertical)
  38  C_TR_BR   TR south exit -> BR north entry (going South, right  vertical)
  39  C_BR_TR   BR north exit -> TR south entry (going North, right  vertical)

Signal groups  (each connector belongs to the DOWNSTREAM signal)
--------------
TL signal: group0=[0, 33] (E-W), group1=[1, 37] (N-S)
TR signal: group0=[8, 32] (E-W), group1=[9, 39] (N-S)
BL signal: group0=[16, 35] (E-W), group1=[17, 36] (N-S)
BR signal: group0=[24, 34] (E-W), group1=[25, 38] (N-S)

Vehicle paths (8 two-intersection corridors, weight 3 each = total 24):
  Top    row  W->E: [0,4,32,12,10]
  Top    row  E->W: [8,13,33,5,2]
  Bottom row  W->E: [16,20,34,28,26]
  Bottom row  E->W: [24,29,35,21,18]
  Left   col  N->S: [1,7,36,23,19]
  Left   col  S->N: [17,22,37,6,3]
  Right  col  N->S: [9,15,38,31,27]
  Right  col  S->N: [25,30,39,14,11]
"""

from TrafficSimulator import Simulation

a = 2        # lane offset from centreline
b = 12       # inner junction radius
length = 40  # external arm length
cx = 60      # intersection centre x-offset
cy = 60      # intersection centre y-offset

# ── TL inner nodes (centre: -cx, -cy) ────────────────────────────────────────
TL_W_R = (-cx-b, -cy+a)    # (-72, -58)
TL_W_L = (-cx-b, -cy-a)    # (-72, -62)
TL_E_L = (-cx+b, -cy+a)    # (-48, -58)
TL_E_R = (-cx+b, -cy-a)    # (-48, -62)
TL_S_R = (-cx+a, -cy+b)    # (-58, -48)
TL_S_L = (-cx-a, -cy+b)    # (-62, -48)
TL_N_R = (-cx-a, -cy-b)    # (-62, -72)
TL_N_L = (-cx+a, -cy-b)    # (-58, -72)

TL_W_R_START = (-cx-b-length, -cy+a)    # (-112, -58)
TL_W_L_START = (-cx-b-length, -cy-a)    # (-112, -62)
TL_N_R_START = (-cx-a, -cy-b-length)    # ( -62,-112)
TL_N_L_START = (-cx+a, -cy-b-length)    # ( -58,-112)

# ── TR inner nodes (centre: +cx, -cy) ────────────────────────────────────────
TR_W_R = (cx-b, -cy+a)     # (48, -58)
TR_W_L = (cx-b, -cy-a)     # (48, -62)
TR_E_R = (cx+b, -cy-a)     # (72, -62)
TR_E_L = (cx+b, -cy+a)     # (72, -58)
TR_S_R = (cx+a, -cy+b)     # (62, -48)
TR_S_L = (cx-a, -cy+b)     # (58, -48)
TR_N_R = (cx-a, -cy-b)     # (58, -72)
TR_N_L = (cx+a, -cy-b)     # (62, -72)

TR_E_R_START = (cx+b+length, -cy-a)     # (112, -62)
TR_E_L_START = (cx+b+length, -cy+a)     # (112, -58)
TR_N_R_START = (cx-a, -cy-b-length)     # ( 58,-112)
TR_N_L_START = (cx+a, -cy-b-length)     # ( 62,-112)

# ── BL inner nodes (centre: -cx, +cy) ────────────────────────────────────────
BL_W_R = (-cx-b, cy+a)     # (-72, 62)
BL_W_L = (-cx-b, cy-a)     # (-72, 58)
BL_E_L = (-cx+b, cy+a)     # (-48, 62)
BL_E_R = (-cx+b, cy-a)     # (-48, 58)
BL_S_R = (-cx+a, cy+b)     # (-58, 72)
BL_S_L = (-cx-a, cy+b)     # (-62, 72)
BL_N_R = (-cx-a, cy-b)     # (-62, 48)
BL_N_L = (-cx+a, cy-b)     # (-58, 48)

BL_W_R_START = (-cx-b-length, cy+a)     # (-112, 62)
BL_W_L_START = (-cx-b-length, cy-a)     # (-112, 58)
BL_S_R_START = (-cx+a, cy+b+length)     # ( -58,112)
BL_S_L_START = (-cx-a, cy+b+length)     # ( -62,112)

# ── BR inner nodes (centre: +cx, +cy) ────────────────────────────────────────
BR_W_R = (cx-b, cy+a)      # (48, 62)
BR_W_L = (cx-b, cy-a)      # (48, 58)
BR_E_L = (cx+b, cy+a)      # (72, 62)
BR_E_R = (cx+b, cy-a)      # (72, 58)
BR_S_R = (cx+a, cy+b)      # (62, 72)
BR_S_L = (cx-a, cy+b)      # (58, 72)
BR_N_R = (cx-a, cy-b)      # (58, 48)
BR_N_L = (cx+a, cy-b)      # (62, 48)

BR_E_R_START = (cx+b+length, cy-a)      # (112, 58)
BR_E_L_START = (cx+b+length, cy+a)      # (112, 62)
BR_S_R_START = (cx+a, cy+b+length)      # ( 62,112)
BR_S_L_START = (cx-a, cy+b+length)      # ( 58,112)

# ── Road definitions ──────────────────────────────────────────────────────────
ROADS = [
    # TL (0-7)
    (TL_W_R_START, TL_W_R),    #  0  TL_W_IN
    (TL_N_R_START, TL_N_R),    #  1  TL_N_IN
    (TL_W_L,  TL_W_L_START),   #  2  TL_W_OUT
    (TL_N_L,  TL_N_L_START),   #  3  TL_N_OUT
    (TL_W_R,  TL_E_L),         #  4  TL_WE_STR (East)
    (TL_E_R,  TL_W_L),         #  5  TL_EW_STR (West)
    (TL_S_R,  TL_N_L),         #  6  TL_SN_STR (North)
    (TL_N_R,  TL_S_L),         #  7  TL_NS_STR (South)

    # TR (8-15)
    (TR_E_R_START, TR_E_R),    #  8  TR_E_IN
    (TR_N_R_START, TR_N_R),    #  9  TR_N_IN
    (TR_E_L,  TR_E_L_START),   # 10  TR_E_OUT
    (TR_N_L,  TR_N_L_START),   # 11  TR_N_OUT
    (TR_W_R,  TR_E_L),         # 12  TR_WE_STR (East)
    (TR_E_R,  TR_W_L),         # 13  TR_EW_STR (West)
    (TR_S_R,  TR_N_L),         # 14  TR_SN_STR (North)
    (TR_N_R,  TR_S_L),         # 15  TR_NS_STR (South)

    # BL (16-23)
    (BL_W_R_START, BL_W_R),    # 16  BL_W_IN
    (BL_S_R_START, BL_S_R),    # 17  BL_S_IN
    (BL_W_L,  BL_W_L_START),   # 18  BL_W_OUT
    (BL_S_L,  BL_S_L_START),   # 19  BL_S_OUT
    (BL_W_R,  BL_E_L),         # 20  BL_WE_STR (East)
    (BL_E_R,  BL_W_L),         # 21  BL_EW_STR (West)
    (BL_S_R,  BL_N_L),         # 22  BL_SN_STR (North)
    (BL_N_R,  BL_S_L),         # 23  BL_NS_STR (South)

    # BR (24-31)
    (BR_E_R_START, BR_E_R),    # 24  BR_E_IN
    (BR_S_R_START, BR_S_R),    # 25  BR_S_IN
    (BR_E_L,  BR_E_L_START),   # 26  BR_E_OUT
    (BR_S_L,  BR_S_L_START),   # 27  BR_S_OUT
    (BR_W_R,  BR_E_L),         # 28  BR_WE_STR (East)
    (BR_E_R,  BR_W_L),         # 29  BR_EW_STR (West)
    (BR_S_R,  BR_N_L),         # 30  BR_SN_STR (North)
    (BR_N_R,  BR_S_L),         # 31  BR_NS_STR (South)

    # Horizontal connectors (32-35)
    (TL_E_L,  TR_W_R),         # 32  C_TL_TR  East  top
    (TR_W_L,  TL_E_R),         # 33  C_TR_TL  West  top
    (BL_E_L,  BR_W_R),         # 34  C_BL_BR  East  bottom
    (BR_W_L,  BL_E_R),         # 35  C_BR_BL  West  bottom

    # Vertical connectors (36-39)
    (TL_S_L,  BL_N_R),         # 36  C_TL_BL  South left
    (BL_N_L,  TL_S_R),         # 37  C_BL_TL  North left
    (TR_S_L,  BR_N_R),         # 38  C_TR_BR  South right
    (BR_N_L,  TR_S_R),         # 39  C_BR_TR  North right
]

# ── Vehicle generator ─────────────────────────────────────────────────────────
VEHICLE_RATE = 30   # vehicles / minute

PATHS = [
    # Top horizontal corridor
    [3, [0,  4, 32, 12, 10]],   # W_TL -> E_TR  (through TL then TR going east)
    [3, [8, 13, 33,  5,  2]],   # E_TR -> W_TL  (through TR then TL going west)
    # Bottom horizontal corridor
    [3, [16, 20, 34, 28, 26]],  # W_BL -> E_BR  (through BL then BR going east)
    [3, [24, 29, 35, 21, 18]],  # E_BR -> W_BL  (through BR then BL going west)
    # Left vertical corridor
    [3, [1,  7, 36, 23, 19]],   # N_TL -> S_BL  (through TL then BL going south)
    [3, [17, 22, 37,  6,  3]],  # S_BL -> N_TL  (through BL then TL going north)
    # Right vertical corridor
    [3, [9, 15, 38, 31, 27]],   # N_TR -> S_BR  (through TR then BR going south)
    [3, [25, 30, 39, 14, 11]],  # S_BR -> N_TR  (through BR then TR going north)
]
# Total weight = 24  ->  each node carries 4 routes * weight 3 = 12 units

# ── Signal configuration ──────────────────────────────────────────────────────
TL_SIGNAL_ROADS = [[0, 33], [1, 37]]    # group0=E-W, group1=N-S
TR_SIGNAL_ROADS = [[8, 32], [9, 39]]
BL_SIGNAL_ROADS = [[16, 35], [17, 36]]
BR_SIGNAL_ROADS = [[24, 34], [25, 38]]

CYCLE        = [(False, True), (False, False), (True, False), (False, False)]
SLOW_DIST    = 50
SLOW_FACTOR  = 0.4
STOP_DIST    = 15

# ── Metadata for agents ───────────────────────────────────────────────────────
TL_THROUGH_ROADS = {4, 5, 6, 7}
TR_THROUGH_ROADS = {12, 13, 14, 15}
BL_THROUGH_ROADS = {20, 21, 22, 23}
BR_THROUGH_ROADS = {28, 29, 30, 31}

CONNECTOR_LENGTH = 96.0   # distance between adjacent intersection centres

# Intersection centres
TL_CENTRE = (-cx, -cy)
TR_CENTRE = ( cx, -cy)
BL_CENTRE = (-cx,  cy)
BR_CENTRE = ( cx,  cy)

# ── Collision detection ───────────────────────────────────────────────────────
INTERSECTIONS_DICT = {
    # TL: E-W straights cross N-S straights
    4:  {6, 7},   5:  {6, 7},   6:  {4, 5},   7:  {4, 5},
    # TR
    12: {14, 15}, 13: {14, 15}, 14: {12, 13}, 15: {12, 13},
    # BL
    20: {22, 23}, 21: {22, 23}, 22: {20, 21}, 23: {20, 21},
    # BR
    28: {30, 31}, 29: {30, 31}, 30: {28, 29}, 31: {28, 29},
}


def grid_intersection_setup(max_gen=None) -> Simulation:
    sim = Simulation(max_gen)
    sim.add_roads(ROADS)
    sim.add_generator(VEHICLE_RATE, PATHS)
    sim.add_traffic_signal(TL_SIGNAL_ROADS, CYCLE, SLOW_DIST, SLOW_FACTOR, STOP_DIST)
    sim.add_traffic_signal(TR_SIGNAL_ROADS, CYCLE, SLOW_DIST, SLOW_FACTOR, STOP_DIST)
    sim.add_traffic_signal(BL_SIGNAL_ROADS, CYCLE, SLOW_DIST, SLOW_FACTOR, STOP_DIST)
    sim.add_traffic_signal(BR_SIGNAL_ROADS, CYCLE, SLOW_DIST, SLOW_FACTOR, STOP_DIST)
    sim.add_intersections(INTERSECTIONS_DICT)
    return sim
