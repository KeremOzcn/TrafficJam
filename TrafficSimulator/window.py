import math

import numpy as np
import pygame
from pygame.draw import polygon

# ── Layout ────────────────────────────────────────────────────────────────────
SIM_WIDTH     = 900
PANEL_WIDTH   = 460
TOTAL_WIDTH   = SIM_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = 780
BOTTOM_H      = 60
SIM_HEIGHT    = WINDOW_HEIGHT - BOTTOM_H   # 720

# ── Palette ───────────────────────────────────────────────────────────────────
CLR_SIM_BG    = (12,  14,  28)
CLR_ROAD      = (38,  40,  58)
CLR_PANEL_BG  = (10,  18,  42)
CLR_CARD      = (18,  30,  62)
CLR_CARD_HDR  = (22,  40,  80)
CLR_ACCENT    = (74, 158, 255)
CLR_WHITE     = (255, 255, 255)
CLR_GREY      = (140, 150, 175)
CLR_GREEN     = (  0, 200,  83)
CLR_RED       = (255,  61,  61)
CLR_YELLOW    = (255, 214,   0)
CLR_BOTTOM    = (  8,  14,  32)
CLR_BTN_STOP  = (180,  40,  40)
CLR_SLIDER_BG = ( 30,  45,  90)
CLR_SLIDER_FG = ( 74, 158, 255)
CLR_LOG_BG    = (  8,  16,  38)
CLR_LOG_INFO  = (200, 205, 220)
CLR_LOG_GREEN = ( 20, 210,  90)
CLR_LOG_RED   = (255, 100, 100)
CLR_LOG_YELLOW= (255, 200,  60)

METHOD_LABELS = {
    'fc':        'Fixed Cycle',
    'lqf':       'Longest Queue First',
    'qlearning': 'Q-Learning',
    'search':    'Genetic Algorithm',
    'mcts':      'MCTS',
    'logic':     'Propositional Logic',
    'math':      'Math Model (LA + Prob)',
    'network':   'Network Math (Jackson)',
    'grid':      'Grid Network (2x2)',
    '':          'Unknown',
}

SPEED_MIN      = 0.25
SPEED_MAX      = 8.0
MAX_LOG_ENTRIES = 300


class Window:
    def __init__(self, simulation):
        self._sim = simulation
        self.closed: bool = False

        self._screen = pygame.display.set_mode((TOTAL_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('AI Traffic Lights Controller')
        pygame.display.flip()
        pygame.font.init()

        _f = 'Lucida Console'
        self._font_sm   = pygame.font.SysFont(_f, 13)
        self._font_md   = pygame.font.SysFont(_f, 15)
        self._font_lg   = pygame.font.SysFont(_f, 20)
        self._font_xl   = pygame.font.SysFont(_f, 22, bold=True)
        self._font_hdr  = pygame.font.SysFont(_f, 13, bold=True)
        self._text_font = self._font_md   # backward-compat alias

        self._zoom       = 4.0
        self._offset     = (0, 0)
        self._mouse_last = (0, 0)
        self._mouse_down = False

        # Episode wait-time history (kept for external callers)
        self.episode_history: list = []

        # Terminal log: list of (text, colour)
        self._log: list = []

        # Auto-log state tracking
        self._last_log_t: float          = -999.0
        self._last_signal_states: dict   = {}
        self._last_n_gen: int            = 0

        # Button / slider rects — set properly each frame in _draw_bottom
        self._btn_play  = pygame.Rect(0, 0, 0, 0)
        self._btn_pause = pygame.Rect(0, 0, 0, 0)
        self._btn_stop  = pygame.Rect(0, 0, 0, 0)
        self._slider_x  = 0
        self._slider_w  = 1
        self._slider_y  = 0
        self._slider_dragging = False

    # ── Public helpers ────────────────────────────────────────────────────────

    def record_episode(self, wait_time: float) -> None:
        self.episode_history.append(wait_time)
        if len(self.episode_history) > 20:
            self.episode_history.pop(0)
        ep = self._sim.dashboard_info.get('episode', '?')
        self.log_entry(f'[INFO] Episode {ep} done — avg wait: {wait_time:.2f} s',
                       CLR_LOG_GREEN)

    def log_entry(self, text: str, color=None) -> None:
        """Append a line to the terminal log panel."""
        if color is None:
            color = CLR_LOG_INFO
        self._log.append((text, color))
        if len(self._log) > MAX_LOG_ENTRIES:
            self._log.pop(0)

    # ── Main update loop ──────────────────────────────────────────────────────

    def update(self) -> None:
        self._auto_log()
        self._draw()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._sim.paused = not self._sim.paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._sim.speed_factor = min(SPEED_MAX,
                        round(self._sim.speed_factor * 2, 2))
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._sim.speed_factor = max(SPEED_MIN,
                        round(self._sim.speed_factor / 2, 2))

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if event.button == pygame.BUTTON_LEFT:
                    if my >= SIM_HEIGHT:
                        self._handle_bottom_click(mx, my)
                    elif not self._is_on_slider(mx, my) and mx < SIM_WIDTH:
                        x0, y0 = self._offset
                        self._mouse_last = (mx - x0 * self._zoom,
                                            my - y0 * self._zoom)
                        self._mouse_down = True
                    if self._is_on_slider(mx, my):
                        self._slider_dragging = True
                        self._update_speed_from_slider(mx)
                if event.button == pygame.BUTTON_WHEELUP and mx < SIM_WIDTH:
                    self._zoom = min(20.0, self._zoom * 1.1)
                if event.button == pygame.BUTTON_WHEELDOWN and mx < SIM_WIDTH:
                    self._zoom = max(1.0, self._zoom / 1.1)

            elif event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                if self._mouse_down:
                    x1, y1 = self._mouse_last
                    self._offset = ((mx - x1) / self._zoom,
                                    (my - y1) / self._zoom)
                if self._slider_dragging:
                    self._update_speed_from_slider(mx)

            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_down = False
                self._slider_dragging = False

    # ── Auto-log ──────────────────────────────────────────────────────────────

    def _auto_log(self) -> None:
        t = self._sim.t
        # Sim-time heartbeat every 5 s
        if t - self._last_log_t >= 5.0:
            self._last_log_t = t
            self.log_entry(f'[INFO] Sim time: {t:.1f} s', CLR_LOG_INFO)

        # Signal state changes
        n_sigs = len(self._sim.traffic_signals)
        sig_labels = (['TL', 'TR', 'BL', 'BR'] if n_sigs > 2
                      else ['Left', 'Right'][:n_sigs])
        for i, sig in enumerate(self._sim.traffic_signals):
            cyc = tuple(sig.current_cycle)
            if cyc != self._last_signal_states.get(i):
                self._last_signal_states[i] = cyc
                lbl = sig_labels[i] if i < len(sig_labels) else str(i)
                if cyc == (False, False):
                    self.log_entry(f'[INFO] {lbl}: YELLOW phase', CLR_LOG_YELLOW)
                else:
                    ew = 'GREEN' if cyc[0] else 'RED'
                    ns = 'GREEN' if cyc[1] else 'RED'
                    col = CLR_LOG_GREEN if cyc[0] else CLR_LOG_YELLOW
                    self.log_entry(f'[INFO] {lbl}: W/E {ew},  S/N {ns}', col)

        # Vehicle generation (log every 5 new vehicles)
        gen = self._sim.n_vehicles_generated
        if gen != self._last_n_gen:
            self._last_n_gen = gen
            if gen % 5 == 0:
                self.log_entry(f'[INFO] Vehicles generated: {gen}', CLR_LOG_GREEN)

    # ── Bottom-bar interaction ────────────────────────────────────────────────

    def _is_on_slider(self, mx, my) -> bool:
        return (self._slider_w > 1
                and self._slider_x <= mx <= self._slider_x + self._slider_w
                and abs(my - self._slider_y) < 16)

    def _update_speed_from_slider(self, mx: int) -> None:
        frac = (mx - self._slider_x) / max(self._slider_w, 1)
        frac = max(0.0, min(1.0, frac))
        raw  = math.exp(math.log(SPEED_MIN) + frac * (math.log(SPEED_MAX) - math.log(SPEED_MIN)))
        snaps = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        self._sim.speed_factor = min(snaps, key=lambda s: abs(s - raw))

    def _handle_bottom_click(self, mx, my) -> None:
        if self._btn_play.collidepoint(mx, my):
            self._sim.paused = False
        elif self._btn_pause.collidepoint(mx, my):
            self._sim.paused = True
        elif self._btn_stop.collidepoint(mx, my):
            self.closed = True

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _convert(self, x, y=None):
        if isinstance(x, list):
            return [self._convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self._convert(*x)
        return (int(SIM_WIDTH / 2 + (x + self._offset[0]) * self._zoom),
                int(SIM_HEIGHT / 2 + (y + self._offset[1]) * self._zoom))

    def _inverse_convert(self, x, y=None):
        if isinstance(x, list):
            return [self._inverse_convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self._inverse_convert(*x)
        return (int(-self._offset[0] + (x - SIM_WIDTH / 2) / self._zoom),
                int(-self._offset[1] + (y - SIM_HEIGHT / 2) / self._zoom))

    # ── Simulation drawing ────────────────────────────────────────────────────

    def _rotated_box(self, pos, size, angle=None, cos=None, sin=None,
                     centered=True, color=(0, 0, 255)):
        def vertex(e1, e2):
            return (x + (e1 * l * cos + e2 * h * sin) / 2,
                    y + (e1 * l * sin - e2 * h * cos) / 2)
        x, y = pos
        l, h = size
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        if centered:
            pts = self._convert(
                [vertex(*e) for e in [(-1, -1), (-1, 1), (1, 1), (1, -1)]])
        else:
            pts = self._convert(
                [vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]])
        polygon(self._screen, color, pts)

    def _draw_arrow(self, pos, size, angle=None, cos=None, sin=None,
                    color=(60, 65, 90)):
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        self._rotated_box(pos, size,
                          cos=(cos - sin) / np.sqrt(2),
                          sin=(cos + sin) / np.sqrt(2),
                          color=color, centered=False)
        self._rotated_box(pos, size,
                          cos=(cos + sin) / np.sqrt(2),
                          sin=(sin - cos) / np.sqrt(2),
                          color=color, centered=False)

    def _draw_roads(self) -> None:
        for road in self._sim.roads:
            cos, sin = road.angle_cos, road.angle_sin

            # ── Road surface ───────────────────────────────────────────────
            self._rotated_box(road.start, (road.length, 3.8),
                              cos=cos, sin=sin, color=CLR_ROAD, centered=False)

            if road.length > 6:
                # ── Yellow edge lines ──────────────────────────────────────
                edge = 1.72
                for sign in (-1, 1):
                    es = (road.start[0] + sign * sin * edge,
                          road.start[1] - sign * cos * edge)
                    self._rotated_box(es, (road.length, 0.20),
                                      cos=cos, sin=sin,
                                      color=(175, 138, 18), centered=False)

                # ── White dashed centre line ───────────────────────────────
                dash, gap = 2.6, 2.0
                period    = dash + gap
                t         = dash / 2
                while t + dash / 2 < road.length:
                    cx = road.start[0] + cos * t
                    cy = road.start[1] + sin * t
                    self._rotated_box((cx, cy), (dash, 0.18),
                                      cos=cos, sin=sin,
                                      color=(210, 212, 220), centered=True)
                    t += period

                # ── Direction arrows ───────────────────────────────────────
                for i in np.arange(-0.5 * road.length, 0.5 * road.length, 10):
                    pos = (road.start[0] + (road.length / 2 + i + 3) * cos,
                           road.start[1] + (road.length / 2 + i + 3) * sin)
                    self._draw_arrow(pos, (-1.25, 0.2), cos=cos, sin=sin)

    def _draw_vehicle(self, vehicle, road) -> None:
        l, h = vehicle.length, vehicle.width
        sin, cos = road.angle_sin, road.angle_cos
        x = road.start[0] + cos * vehicle.x
        y = road.start[1] + sin * vehicle.x

        # Drop shadow
        self._rotated_box((x + 0.18, y + 0.18), (l, h), cos=cos, sin=sin,
                          centered=True, color=(4, 4, 14))
        # Body
        self._rotated_box((x, y), (l, h), cos=cos, sin=sin,
                          centered=True, color=(45, 108, 220))
        # Roof / windshield (centred, smaller)
        self._rotated_box((x, y), (l * 0.42, h * 0.58), cos=cos, sin=sin,
                          centered=True, color=(80, 150, 255))
        # Front headlights (two small yellow dots at front edge)
        for side in (-0.28, 0.28):
            hx = x + cos * l * 0.46 - sin * h * side
            hy = y + sin * l * 0.46 + cos * h * side
            sx, sy = self._convert(hx, hy)
            r = max(1, int(self._zoom * 0.28))
            pygame.draw.circle(self._screen, (255, 240, 160), (sx, sy), r)
        # Rear taillights (two small red dots at rear edge)
        for side in (-0.28, 0.28):
            hx = x - cos * l * 0.46 - sin * h * side
            hy = y - sin * l * 0.46 + cos * h * side
            sx, sy = self._convert(hx, hy)
            r = max(1, int(self._zoom * 0.28))
            pygame.draw.circle(self._screen, (220, 50, 50), (sx, sy), r)

    def _draw_vehicles(self) -> None:
        for i in self._sim.non_empty_roads:
            road = self._sim.roads[i]
            for vehicle in road.vehicles:
                self._draw_vehicle(vehicle, road)

    def _draw_signals(self) -> None:
        for signal in self._sim.traffic_signals:
            for i in range(len(signal.roads)):
                if signal.current_cycle == (False, False):
                    prev = signal.cycle[signal.current_cycle_index - 1]
                    active = CLR_YELLOW if prev[i] else CLR_RED
                else:
                    active = CLR_GREEN if signal.current_cycle[i] else CLR_RED
                for road in signal.roads[i]:
                    self._draw_traffic_light(road, active)

    # which of the 3 bulbs (0=red,1=yellow,2=green) is the active one
    _BULB_IDX = {
        id(CLR_RED):    0,
        id(CLR_YELLOW): 1,
        id(CLR_GREEN):  2,
    }
    _BULB_COLORS_LIT = [CLR_RED, CLR_YELLOW, CLR_GREEN]
    _BULB_COLORS_DIM = [(55, 14, 14), (55, 50, 14), (14, 55, 22)]

    def _draw_traffic_light(self, road, active_color) -> None:
        z = self._zoom
        # Position near road-end, offset to right-hand side
        a   = 0.10
        bx  = (1 - a) * road.end[0] + a * road.start[0]
        by  = (1 - a) * road.end[1] + a * road.start[1]
        bx += road.angle_sin * 2.4
        by -= road.angle_cos * 2.4
        sx, sy = self._convert(bx, by)

        # Scale dimensions with zoom (clamped for readability)
        pole_w = max(2, int(z * 0.30))
        pole_h = max(8, int(z * 3.2))
        box_w  = max(6, int(z * 1.20))
        box_h  = max(14, int(z * 3.80))
        bulb_r = max(1, int(z * 0.36))

        # Pole
        pygame.draw.rect(self._screen, (50, 52, 60),
                         (sx - pole_w // 2, sy - pole_h, pole_w, pole_h))

        # Housing box (dark, above pole)
        hx = sx - box_w // 2
        hy = sy - pole_h - box_h
        pygame.draw.rect(self._screen, (14, 15, 20),
                         (hx, hy, box_w, box_h), border_radius=2)
        pygame.draw.rect(self._screen, (38, 40, 50),
                         (hx, hy, box_w, box_h), 1, border_radius=2)

        # 3 bulbs: red (top) → yellow (middle) → green (bottom)
        lit_idx = (0 if active_color is CLR_RED else
                   1 if active_color is CLR_YELLOW else 2)
        for k in range(3):
            y_bulb = hy + box_h * (2 * k + 1) // 6
            if k == lit_idx:
                # Glow ring behind lit bulb
                gc = tuple(min(255, v // 2) for v in self._BULB_COLORS_LIT[k])
                pygame.draw.circle(self._screen, gc, (sx, y_bulb), bulb_r + 2)
                pygame.draw.circle(self._screen, self._BULB_COLORS_LIT[k],
                                   (sx, y_bulb), bulb_r)
            else:
                pygame.draw.circle(self._screen, self._BULB_COLORS_DIM[k],
                                   (sx, y_bulb), bulb_r)

    def _draw_status(self) -> None:
        surf = self._font_sm.render(f't = {self._sim.t:.1f} s', True, (160, 170, 200))
        self._screen.blit(surf, (10, 8))
        if self._sim.paused:
            ps = self._font_lg.render('  PAUSED  ', True, CLR_WHITE, (150, 40, 40))
            self._screen.blit(ps, (SIM_WIDTH // 2 - ps.get_width() // 2, 10))

    # ── Right panel ───────────────────────────────────────────────────────────

    def _draw_panel(self) -> None:
        px0   = SIM_WIDTH
        sim   = self._sim
        info  = sim.dashboard_info
        pw    = PANEL_WIDTH
        card_w = pw - 28
        px    = px0 + 14

        # Background + left border
        pygame.draw.rect(self._screen, CLR_PANEL_BG, (px0, 0, pw, WINDOW_HEIGHT))
        pygame.draw.line(self._screen, CLR_ACCENT, (px0, 0), (px0, WINDOW_HEIGHT), 2)

        y = 14

        # ── Title ──────────────────────────────────────────────────────────
        surf = self._font_hdr.render('AI TRAFFIC CONTROLLER', True, CLR_ACCENT)
        self._screen.blit(surf, (px, y)); y += 18
        method_key = info.get('method', '')
        label = METHOD_LABELS.get(method_key, method_key.upper())
        surf = self._font_sm.render(f'Method: {label}', True, CLR_GREY)
        self._screen.blit(surf, (px, y)); y += 16
        pygame.draw.line(self._screen, CLR_CARD_HDR,
                         (px0 + 5, y), (px0 + pw - 5, y), 1)
        y += 8

        # ── Episode card ───────────────────────────────────────────────────
        ep    = info.get('episode', 0)
        total = info.get('total_episodes', 1)
        self._draw_card(px - 4, y, card_w, 72)
        ep_surf = self._font_xl.render(f'Episode  {ep} / {total}', True, CLR_WHITE)
        self._screen.blit(ep_surf, (px + 4, y + 8))
        bar_x, bar_y = px + 4, y + 46
        bar_bw, bar_h = card_w - 8, 14
        frac = ep / max(total, 1)
        pygame.draw.rect(self._screen, CLR_CARD_HDR,
                         (bar_x, bar_y, bar_bw, bar_h), border_radius=6)
        if frac > 0:
            pygame.draw.rect(self._screen, CLR_GREEN,
                             (bar_x, bar_y, int(bar_bw * frac), bar_h),
                             border_radius=6)
        y += 80

        # ── Live Stats card ────────────────────────────────────────────────
        self._draw_card(px - 4, y, card_w, 108)
        self._draw_card_header(px - 4, y, card_w, 'LIVE STATS')
        y += 26
        rows = [
            ('Avg Wait Time:', f'{sim.current_average_wait_time:.2f} s', CLR_WHITE),
            ('Total Collisions:', str(info.get('collisions', 0)),
             CLR_RED if info.get('collisions', 0) else CLR_WHITE),
            ('Active Vehicles:', str(sim.n_vehicles_on_map), CLR_WHITE),
        ]
        for lbl, val, vcol in rows:
            ls = self._font_sm.render(lbl, True, CLR_GREY)
            vs = self._font_sm.render(val, True, vcol)
            self._screen.blit(ls, (px + 4, y))
            self._screen.blit(vs, (px + card_w - vs.get_width() - 10, y))
            y += 24
        y += 16   # bottom card padding

        # ── Terminal log card (fills remaining height) ─────────────────────
        log_card_h = SIM_HEIGHT - y - 10
        if log_card_h > 60:
            self._draw_card(px - 4, y, card_w, log_card_h)
            self._draw_card_header(px - 4, y, card_w, 'TERMINAL LOG')
            log_y = y + 26
            log_h = log_card_h - 26
            pygame.draw.rect(self._screen, CLR_LOG_BG,
                             (px - 4, log_y, card_w, log_h))

            line_h   = 16
            max_vis  = log_h // line_h
            visible  = self._log[-max_vis:]
            max_chars = (card_w - 12) // 7
            for j, (msg, col) in enumerate(visible):
                ly = log_y + j * line_h
                if ly + line_h > log_y + log_h:
                    break
                if len(msg) > max_chars:
                    msg = msg[:max_chars - 1] + '\u2026'
                s = self._font_sm.render(msg, True, col)
                self._screen.blit(s, (px + 2, ly))

    def _draw_card(self, x, y, w, h) -> None:
        pygame.draw.rect(self._screen, CLR_CARD, (x, y, w, h), border_radius=8)

    def _draw_card_header(self, x, y, w, title: str) -> None:
        pygame.draw.rect(self._screen, CLR_CARD_HDR, (x, y, w, 22), border_radius=8)
        surf = self._font_hdr.render(title, True, CLR_ACCENT)
        self._screen.blit(surf, (x + 8, y + 4))

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _draw_bottom(self) -> None:
        by = SIM_HEIGHT
        bh = BOTTOM_H
        bw = TOTAL_WIDTH

        pygame.draw.rect(self._screen, CLR_BOTTOM, (0, by, bw, bh))
        pygame.draw.line(self._screen, CLR_ACCENT, (0, by), (bw, by), 1)

        btn_w, btn_h = 44, 36
        btn_y = by + (bh - btn_h) // 2

        # ── Play button ────────────────────────────────────────────────────
        self._btn_play = pygame.Rect(14, btn_y, btn_w, btn_h)
        play_col = CLR_GREEN if not self._sim.paused else (30, 80, 50)
        pygame.draw.rect(self._screen, play_col, self._btn_play, border_radius=6)
        cx, cy = self._btn_play.centerx, self._btn_play.centery
        pygame.draw.polygon(self._screen, CLR_WHITE,
                            [(cx - 9, cy - 10), (cx - 9, cy + 10), (cx + 11, cy)])

        # ── Pause button ───────────────────────────────────────────────────
        self._btn_pause = pygame.Rect(64, btn_y, btn_w, btn_h)
        pau_col = CLR_ACCENT if self._sim.paused else (35, 50, 100)
        pygame.draw.rect(self._screen, pau_col, self._btn_pause, border_radius=6)
        cx, cy = self._btn_pause.centerx, self._btn_pause.centery
        pygame.draw.rect(self._screen, CLR_WHITE, (cx - 9, cy - 9, 7, 18))
        pygame.draw.rect(self._screen, CLR_WHITE, (cx + 2,  cy - 9, 7, 18))

        # ── Speed label + slider ───────────────────────────────────────────
        spd = self._sim.speed_factor
        lbl = self._font_sm.render(f'Speed: {spd:.2g}x', True, CLR_GREY)
        lbl_x = 118
        self._screen.blit(lbl, (lbl_x, by + (bh - lbl.get_height()) // 2))

        sl_x = lbl_x + lbl.get_width() + 14
        sl_w = bw - 190 - sl_x
        sl_y = by + bh // 2
        sl_h = 6

        self._slider_x = sl_x
        self._slider_w = sl_w
        self._slider_y = sl_y

        frac = ((math.log(max(spd, SPEED_MIN)) - math.log(SPEED_MIN))
                / (math.log(SPEED_MAX) - math.log(SPEED_MIN)))
        frac = max(0.0, min(1.0, frac))

        # Track background
        pygame.draw.rect(self._screen, CLR_SLIDER_BG,
                         (sl_x, sl_y - sl_h // 2, sl_w, sl_h), border_radius=3)
        # Fill
        fw = int(sl_w * frac)
        if fw > 0:
            pygame.draw.rect(self._screen, CLR_SLIDER_FG,
                             (sl_x, sl_y - sl_h // 2, fw, sl_h), border_radius=3)
        # Handle
        hx = sl_x + fw
        pygame.draw.circle(self._screen, CLR_WHITE, (hx, sl_y), 9)
        pygame.draw.circle(self._screen, CLR_SLIDER_FG, (hx, sl_y), 7)

        # "8x" end label
        lbl8 = self._font_sm.render('8x', True, CLR_GREY)
        self._screen.blit(lbl8, (sl_x + sl_w + 6,
                                  by + (bh - lbl8.get_height()) // 2))

        # ── Stop button ────────────────────────────────────────────────────
        self._btn_stop = pygame.Rect(bw - 160, btn_y, 146, btn_h)
        pygame.draw.rect(self._screen, CLR_BTN_STOP, self._btn_stop, border_radius=6)
        ss = self._font_md.render('Stop Simulation', True, CLR_WHITE)
        self._screen.blit(ss,
                          (self._btn_stop.x + (self._btn_stop.w - ss.get_width()) // 2,
                           self._btn_stop.y + (self._btn_stop.h - ss.get_height()) // 2))

    # ── Master draw ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._screen.fill(CLR_SIM_BG, pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT))
        self._draw_roads()
        self._draw_vehicles()
        self._draw_signals()
        self._draw_status()
        self._draw_panel()
        self._draw_bottom()
