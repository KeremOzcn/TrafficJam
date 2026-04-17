import numpy as np
import pygame
from pygame.draw import polygon


# Simulation viewport width (unchanged)
SIM_WIDTH = 1000
# Right dashboard panel width
PANEL_WIDTH = 280
TOTAL_WIDTH = SIM_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = 630

# Colour palette
CLR_BG          = (235, 235, 235)
CLR_PANEL_BG    = (30,  30,  45)
CLR_PANEL_HDR   = (50,  50,  70)
CLR_ACCENT      = (80, 160, 255)
CLR_WHITE       = (255, 255, 255)
CLR_GREY        = (160, 160, 175)
CLR_GREEN       = (60,  200,  90)
CLR_RED         = (220,  60,  60)
CLR_YELLOW      = (240, 200,  50)
CLR_BAR_1       = (80,  160, 255)
CLR_BAR_2       = (255, 140,  50)
CLR_HIST_BAR    = (100, 180, 255)
CLR_HIST_BEST   = (60,  210,  90)

METHOD_LABELS = {
    'fc':        'Fixed Cycle',
    'lqf':       'Longest Queue First',
    'qlearning': 'Q-Learning',
    'search':    'Genetic Algorithm',
    'mcts':      'MCTS',
    'logic':     'Propositional Logic',
    'math':      'Math Model (LA + Prob)',
    '':          'Unknown',
}


class Window:
    def __init__(self, simulation):
        self._sim = simulation
        self.closed: bool = False

        self._screen = pygame.display.set_mode((TOTAL_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('AI Traffic Lights Controller')
        pygame.display.flip()
        pygame.font.init()

        font = 'Lucida Console'
        self._font_sm  = pygame.font.SysFont(font, 13)
        self._font_md  = pygame.font.SysFont(font, 15)
        self._font_lg  = pygame.font.SysFont(font, 18)
        self._font_hdr = pygame.font.SysFont(font, 17, bold=True)

        # Backward-compat alias used by existing _draw_status code
        self._text_font = self._font_md

        self._zoom = 5
        self._offset = (0, 0)
        self._mouse_last = (0, 0)
        self._mouse_down = False

        # Episode wait-time history for the mini bar chart (max 20 entries)
        self.episode_history: list = []

    # ── Public helpers ────────────────────────────────────────────────────────

    def record_episode(self, wait_time: float) -> None:
        """Call after each episode to update the history bar chart."""
        self.episode_history.append(wait_time)
        if len(self.episode_history) > 20:
            self.episode_history.pop(0)

    # ── Main update loop ──────────────────────────────────────────────────────

    def update(self) -> None:
        self._draw()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._sim.paused = not self._sim.paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._sim.speed_factor = min(8.0,
                        round(self._sim.speed_factor * 2, 2))
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._sim.speed_factor = max(0.25,
                        round(self._sim.speed_factor / 2, 2))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    x, y = pygame.mouse.get_pos()
                    # Only handle pan inside the simulation viewport
                    if x < SIM_WIDTH:
                        x0, y0 = self._offset
                        self._mouse_last = (x - x0 * self._zoom,
                                            y - y0 * self._zoom)
                        self._mouse_down = True
                if event.button == pygame.BUTTON_WHEELUP:
                    self._zoom *= ((self._zoom ** 2 + self._zoom / 4 + 1) /
                                   (self._zoom ** 2 + 1))
                if event.button == pygame.BUTTON_WHEELDOWN:
                    self._zoom *= ((self._zoom ** 2 + 1) /
                                   (self._zoom ** 2 + self._zoom / 4 + 1))
            elif event.type == pygame.MOUSEMOTION:
                if self._mouse_down:
                    x1, y1 = self._mouse_last
                    x2, y2 = pygame.mouse.get_pos()
                    self._offset = ((x2 - x1) / self._zoom,
                                    (y2 - y1) / self._zoom)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_down = False

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _convert(self, x, y=None):
        """Simulation coords → screen coords (within the sim viewport)."""
        if isinstance(x, list):
            return [self._convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self._convert(*x)
        return (int(SIM_WIDTH / 2 + (x + self._offset[0]) * self._zoom),
                int(WINDOW_HEIGHT / 2 + (y + self._offset[1]) * self._zoom))

    def _inverse_convert(self, x, y=None):
        if isinstance(x, list):
            return [self._inverse_convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self._inverse_convert(*x)
        return (int(-self._offset[0] + (x - SIM_WIDTH / 2) / self._zoom),
                int(-self._offset[1] + (y - WINDOW_HEIGHT / 2) / self._zoom))

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
                    color=(150, 150, 190)) -> None:
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
            self._rotated_box(road.start, (road.length, 3.7),
                              cos=road.angle_cos, sin=road.angle_sin,
                              color=(180, 180, 220), centered=False)
            if road.length > 5:
                for i in np.arange(-0.5 * road.length, 0.5 * road.length, 10):
                    pos = (road.start[0] + (road.length / 2 + i + 3) * road.angle_cos,
                           road.start[1] + (road.length / 2 + i + 3) * road.angle_sin)
                    self._draw_arrow(pos, (-1.25, 0.2),
                                     cos=road.angle_cos, sin=road.angle_sin)

    def _draw_vehicle(self, vehicle, road) -> None:
        l, h = vehicle.length, vehicle.width
        sin, cos = road.angle_sin, road.angle_cos
        x = road.start[0] + cos * vehicle.x
        y = road.start[1] + sin * vehicle.x
        self._rotated_box((x, y), (l, h), cos=cos, sin=sin, centered=True)

    def _draw_vehicles(self) -> None:
        for i in self._sim.non_empty_roads:
            road = self._sim.roads[i]
            for vehicle in road.vehicles:
                self._draw_vehicle(vehicle, road)

    def _draw_signals(self) -> None:
        for signal in self._sim.traffic_signals:
            for i in range(len(signal.roads)):
                red, green = (255, 0, 0), (0, 255, 0)
                if signal.current_cycle == (False, False):
                    yellow = (255, 255, 0)
                    color = (yellow if signal.cycle[signal.current_cycle_index - 1][i]
                             else red)
                else:
                    color = green if signal.current_cycle[i] else red
                for road in signal.roads[i]:
                    a = 0
                    position = ((1 - a) * road.end[0] + a * road.start[0],
                                (1 - a) * road.end[1] + a * road.start[1])
                    self._rotated_box(position, (1, 3),
                                      cos=road.angle_cos, sin=road.angle_sin,
                                      color=color)

    def _draw_status(self) -> None:
        """Minimal top-left overlay."""
        def render(text, color=(0, 0, 0), bg=CLR_BG):
            return self._text_font.render(text, True, color, bg)

        self._screen.blit(render(f'Time: {self._sim.t:.1f}'), (10, 20))
        if self._sim.max_gen:
            self._screen.blit(render(f'Max Gen: {self._sim.max_gen}'), (10, 40))

        # Speed / pause indicator
        if self._sim.paused:
            pause_surf = self._font_lg.render('  PAUSED  ', True,
                                              (255, 255, 255), (180, 60, 60))
            self._screen.blit(pause_surf,
                               (SIM_WIDTH // 2 - pause_surf.get_width() // 2, 10))
        else:
            spd = self._sim.speed_factor
            spd_str = f'{spd:.2g}x'
            color = CLR_ACCENT if spd != 1.0 else (80, 80, 80)
            spd_surf = self._font_sm.render(f'Speed: {spd_str}', True, color, CLR_BG)
            self._screen.blit(spd_surf, (10, 58))

    # ── Dashboard panel ───────────────────────────────────────────────────────

    def _panel_rect(self) -> pygame.Rect:
        return pygame.Rect(SIM_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT)

    def _draw_panel(self) -> None:
        px = SIM_WIDTH + 10          # panel left margin
        info = self._sim.dashboard_info
        sim  = self._sim

        # Panel background
        pygame.draw.rect(self._screen, CLR_PANEL_BG, self._panel_rect())
        # Thin separator line
        pygame.draw.line(self._screen, CLR_ACCENT,
                         (SIM_WIDTH, 0), (SIM_WIDTH, WINDOW_HEIGHT), 2)

        y = 14

        # ── Title ──────────────────────────────────────────────────────────
        self._panel_text('AI TRAFFIC CONTROLLER', px, y,
                         self._font_hdr, CLR_ACCENT)
        y += 24
        method_key = info.get('method', '')
        label = METHOD_LABELS.get(method_key, method_key.upper())
        self._panel_text(f'Method: {label}', px, y, self._font_sm, CLR_WHITE)
        y += 20

        self._panel_hline(y); y += 10

        # ── Episode ────────────────────────────────────────────────────────
        ep    = info.get('episode', 0)
        total = info.get('total_episodes', 0)
        self._panel_text('EPISODE', px, y, self._font_hdr, CLR_GREY)
        y += 20
        self._panel_text(f'{ep}  /  {total}', px, y, self._font_lg, CLR_WHITE)
        y += 22
        self._panel_text(f'Sim time:  {sim.t:.1f} s', px, y,
                         self._font_sm, CLR_GREY)
        y += 20

        self._panel_hline(y); y += 10

        # ── Signal state ───────────────────────────────────────────────────
        self._panel_text('SIGNAL STATE', px, y, self._font_hdr, CLR_GREY)
        y += 20
        if sim.traffic_signals:
            sig = sim.traffic_signals[0]
            cyc = sig.current_cycle
            # cyc[0] = W/E green,  cyc[1] = S/N green
            # Yellow phase: both False
            if cyc == (False, False):
                c1 = c2 = CLR_YELLOW
                l1 = l2 = 'YELLOW'
            else:
                c1 = CLR_GREEN if cyc[0] else CLR_RED
                l1 = 'GREEN' if cyc[0] else 'RED'
                c2 = CLR_GREEN if cyc[1] else CLR_RED
                l2 = 'GREEN' if cyc[1] else 'RED'
            pygame.draw.circle(self._screen, c1, (px + 8, y + 7), 7)
            self._panel_text(f'W/E  {l1}', px + 22, y,
                             self._font_sm, CLR_WHITE)
            y += 20
            pygame.draw.circle(self._screen, c2, (px + 8, y + 7), 7)
            self._panel_text(f'S/N  {l2}', px + 22, y,
                             self._font_sm, CLR_WHITE)
            y += 22
        else:
            self._panel_text('No signal data', px, y,
                             self._font_sm, CLR_GREY)
            y += 22

        self._panel_hline(y); y += 10

        # ── Queue lengths ──────────────────────────────────────────────────
        self._panel_text('QUEUE LENGTHS', px, y, self._font_hdr, CLR_GREY)
        y += 20
        q1 = q2 = 0
        if sim.traffic_signals:
            sig = sim.traffic_signals[0]
            q1 = sum(len(road.vehicles) for road in sig.roads[0])
            q2 = sum(len(road.vehicles) for road in sig.roads[1])

        max_q = max(q1, q2, 1)
        bar_max = PANEL_WIDTH - 70

        self._panel_text(f'W/E  {q1:>2}', px, y, self._font_sm, CLR_WHITE)
        y += 16
        bar_w1 = int(q1 / max_q * bar_max)
        pygame.draw.rect(self._screen, CLR_PANEL_HDR,
                         (px, y, bar_max, 10), border_radius=3)
        if bar_w1:
            pygame.draw.rect(self._screen, CLR_BAR_1,
                             (px, y, bar_w1, 10), border_radius=3)
        y += 16

        self._panel_text(f'S/N  {q2:>2}', px, y, self._font_sm, CLR_WHITE)
        y += 16
        bar_w2 = int(q2 / max_q * bar_max)
        pygame.draw.rect(self._screen, CLR_PANEL_HDR,
                         (px, y, bar_max, 10), border_radius=3)
        if bar_w2:
            pygame.draw.rect(self._screen, CLR_BAR_2,
                             (px, y, bar_w2, 10), border_radius=3)
        y += 20

        self._panel_hline(y); y += 10

        # ── Live stats ─────────────────────────────────────────────────────
        self._panel_text('LIVE STATS', px, y, self._font_hdr, CLR_GREY)
        y += 20
        self._panel_text(
            f'Wait time:   {sim.current_average_wait_time:.2f} s',
            px, y, self._font_sm, CLR_WHITE)
        y += 18
        self._panel_text(
            f'Vehicles:    {sim.n_vehicles_on_map} on map',
            px, y, self._font_sm, CLR_WHITE)
        y += 18
        self._panel_text(
            f'Generated:   {sim.n_vehicles_generated}'
            + (f' / {sim.max_gen}' if sim.max_gen else ''),
            px, y, self._font_sm, CLR_WHITE)
        y += 18
        collisions = info.get('collisions', 0)
        col_color = CLR_RED if collisions else CLR_WHITE
        self._panel_text(f'Collisions:  {collisions}',
                         px, y, self._font_sm, col_color)
        y += 22

        self._panel_hline(y); y += 10

        # ── Controls hint ──────────────────────────────────────────────────
        self._panel_text('CONTROLS', px, y, self._font_hdr, CLR_GREY)
        y += 18
        ctrl_color = CLR_YELLOW if self._sim.paused else CLR_GREY
        pause_txt = 'SPACE: RESUME' if self._sim.paused else 'SPACE: Pause'
        self._panel_text(pause_txt, px, y, self._font_sm, ctrl_color)
        y += 16
        spd = self._sim.speed_factor
        self._panel_text(f'+/- : Speed  [{spd:.2g}x]', px, y,
                         self._font_sm, CLR_GREY)
        y += 20

        self._panel_hline(y); y += 10

        # ── Episode history bar chart ──────────────────────────────────────
        self._panel_text('EPISODE HISTORY', px, y, self._font_hdr, CLR_GREY)
        y += 20
        history = self.episode_history
        if history:
            chart_h = 55
            chart_w = PANEL_WIDTH - 20
            best = min(history)
            worst = max(history) if max(history) != best else best + 1
            bar_w = max(4, chart_w // len(history) - 2)

            for idx, wt in enumerate(history):
                bh = int((wt - best) / (worst - best + 1e-9) * chart_h)
                bh = max(4, chart_h - bh)  # taller = better
                bx = px + idx * (bar_w + 2)
                by = y + (chart_h - bh)
                color = CLR_HIST_BEST if wt == best else CLR_HIST_BAR
                pygame.draw.rect(self._screen, color,
                                 (bx, by, bar_w, bh), border_radius=2)

            y += chart_h + 6
            self._panel_text(
                f'Best:  {best:.2f}s   Last: {history[-1]:.2f}s',
                px, y, self._font_sm, CLR_GREY)
            y += 18
        else:
            self._panel_text('No completed episodes yet.',
                             px, y, self._font_sm, CLR_GREY)
            y += 18

    # ── Panel draw helpers ────────────────────────────────────────────────────

    def _panel_text(self, text: str, x: int, y: int,
                    font: pygame.font.Font, color) -> None:
        surf = font.render(text, True, color)
        self._screen.blit(surf, (x, y))

    def _panel_hline(self, y: int) -> None:
        pygame.draw.line(self._screen, CLR_PANEL_HDR,
                         (SIM_WIDTH + 5, y),
                         (SIM_WIDTH + PANEL_WIDTH - 5, y), 1)

    # ── Master draw ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        # Clip drawing to sim viewport so roads don't bleed into panel
        self._screen.fill(CLR_BG,
                          pygame.Rect(0, 0, SIM_WIDTH, WINDOW_HEIGHT))
        self._draw_roads()
        self._draw_vehicles()
        self._draw_signals()
        self._draw_status()
        self._draw_panel()
