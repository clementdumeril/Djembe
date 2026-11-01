"""Listening live: strokes from an audio stream, and the 12-pulse grid they imply (roadmap L1).

A player accompanies on jembe 2 and the model solos. Before it can play, the system
must know, at every moment, where the cycle is going: when the next pulses will fall.

``OnsetStream`` applies the stroke rule of ``djembe.onsets`` to audio arriving block by
block, using only the past plus a fixed look-ahead: spectral flux on 512-sample frames
every 2.5 ms, a local maximum over +/- 7.5 ms above the median of the last 80 ms plus k
times the running median absolute deviation, and an amplitude rise of ``rise_db`` within
15 ms. A stroke is reported about 25 ms after it is played.

``GridTracker`` follows the cycle with a Kalman filter on (start of the current cycle,
cycle duration). The accompaniment strikes known pulses with known small offsets: in
the corpus, jembe 2 plays pulses 1, 3, 4, 6, 7, 9, 10, 12 of the 12, the first of each
beat late by 0.0 to 0.2 pulse and the last slightly early, depending on the piece. A
stroke heard at time t near pulse k is an observation of start + (k + o_k) / 12 x T,
linear in the state, so each stroke corrects both the phase and the tempo. Strokes more
than ``gate`` pulses from every expected pulse are ignored (a fill, noise, the soloist's
own sound). The cycle duration may drift from cycle to cycle (accelerando).

The player starts on the first pulse of a cycle: the jembe-2 strokes alone repeat every
beat, so they give the beat but not which beat starts the cycle. The tracker listens to
a little more than one cycle (``listen`` seconds), then picks the cycle duration that
best explains those strokes (each weighted by its level, so that faint bleed from the
other drums counts little) and replays them through the filter. While tracking, a
stroke quieter than ``level_floor`` times the median level of the strokes it accepted
is ignored: on the corpus' jembe-2 microphone the soloist and the dundun leak in 3 to 10
times weaker than the player's own strokes.

A stroke is accepted if it falls within ``gate`` pulses of an expected one, a window
widened to 2.5 standard deviations of the prediction when the filter is unsure. After
``reacquire_after`` rejected strokes in a row (a sudden tempo change, a fill), the
tracker searches again, over the strokes of the last ``listen`` seconds, for the cycle
start (within half a cycle of where it thought it was) and duration (within 15 % of the
median duration it has tracked so far, so that repeated searches cannot drift) that
best explain them.

``ignore_own`` drops a stroke heard within 15 ms of one the system itself played when it
is no louder than ``own_level`` (the level at which the system's sound comes back into
the microphone, measured once before playing): live, the soloist's sound comes back
through the microphone, and the system knows when it played. A fixed fraction of the
player's level does not work: the soloist and the accompanist often strike together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOP_S, NFFT = 0.0025, 512
J2_PULSES = (0, 2, 3, 5, 6, 8, 9, 11)


class OnsetStream:
    """Stroke onsets from audio pushed block by block; ``push`` returns the newly confirmed ones."""

    def __init__(self, sr: int, k: float = 3.0, rise_db: float = 4.0, min_ioi: float = 0.04, floor: float = 0.05):
        self.sr, self.k, self.rise_db, self.min_ioi, self.floor = sr, k, rise_db, min_ioi, floor
        self.hop = int(round(HOP_S * sr))
        self.win = np.hanning(NFFT)
        self.buf = np.zeros(0, np.float32)       # samples not yet consumed by a frame
        self.t0 = 0                               # absolute sample index of buf[0]
        self.frames = 0
        self.prev_mag = None
        self.flux: list[float] = []
        self.env: list[float] = []                # per-hop peak amplitude
        self.mad = None
        self.last = -1.0
        self.levels: list[float] = []
        self.half = int(round(0.0075 / HOP_S))    # local-maximum half width (frames)
        self.rise = int(round(0.015 / HOP_S))     # frames after the candidate for the rise check
        self.med = int(round(0.080 / HOP_S))

    def push(self, block: np.ndarray) -> list[tuple[float, float]]:
        """Feed samples; returns [(time s, level)] of the strokes confirmed by this block."""
        self.buf = np.concatenate([self.buf, np.asarray(block, np.float32)])
        out = []
        while len(self.buf) >= NFFT:
            frame = self.buf[:NFFT]
            mag = np.log1p(1000 * np.abs(np.fft.rfft(frame * self.win)))
            self.flux.append(0.0 if self.prev_mag is None else float(np.maximum(mag - self.prev_mag, 0).sum()))
            self.env.append(float(np.abs(frame[NFFT // 2 - self.hop // 2:NFFT // 2 + self.hop // 2 + 1]).max()))
            self.prev_mag = mag
            self.buf = self.buf[self.hop:]
            self.t0 += self.hop
            self.frames += 1
            out += self._decide()
        return out

    def _time(self, j: int) -> float:
        return (j * self.hop + NFFT / 2) / self.sr

    def _decide(self) -> list[tuple[float, float]]:
        j = self.frames - 1 - max(self.half, self.rise)          # the frame we can now judge
        if j < self.med:
            return []
        f = self.flux
        window = f[j - self.half:j + self.half + 1]
        recent = np.asarray(f[max(0, j - 4 * self.med):j + 1])
        dev = float(np.median(np.abs(recent - np.median(recent)))) + 1e-9
        self.mad = dev if self.mad is None else 0.995 * self.mad + 0.005 * dev
        thr = float(np.median(f[j - self.med:j + 1])) + self.k * self.mad
        if f[j] < thr or f[j] < max(window):
            return []
        before = np.mean(self.env[max(0, j - 3):j - 1]) + 1e-9
        after = max(self.env[j:j + self.rise + 1])
        if 20 * np.log10(after / before) < self.rise_db:
            return []
        t = self._time(j)
        if t - self.last < self.min_ioi:
            return []
        typical = np.median(self.levels[-50:]) if len(self.levels) >= 5 else after
        if after < self.floor * typical:
            return []
        self.last = t
        self.levels.append(after)
        return [(t, after)]


@dataclass
class GridTracker:
    """Kalman filter on (start of the current cycle [s], cycle duration [s])."""
    offsets: dict                          # expected offset (fraction of a pulse) of the accompaniment per pulse
    obs_sd: float = 0.06                   # spread of the player's offsets around them, fraction of a pulse
    tempo_drift: float = 0.04              # cycle-to-cycle change of the duration [s]
    gate: float = 0.45                     # strokes further than this from every expected pulse are ignored
    level_floor: float = 0.35              # strokes quieter than this x the median accepted level are ignored
    listen: float = 2.8                    # seconds of listening before the first estimate
    t_range: tuple = (1.0, 2.6)            # plausible cycle durations [s]
    reacquire_after: int = 4
    x: np.ndarray | None = None            # state [start, T]
    P: np.ndarray | None = None
    cycle: int = 0
    log: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    recent: list = field(default_factory=list)
    misses: int = 0
    n_reacquired: int = 0
    tempi: list = field(default_factory=list)
    own: list = field(default_factory=list)          # times of the strokes the system played
    own_level: float = 0.0                            # echo level of the system's strokes (calibrated)

    def start_from(self, strokes: list[tuple[float, float]]) -> None:
        """First stroke on pulse 0; the cycle duration that best explains the strokes heard so far."""
        t0, l0 = strokes[0]
        o0 = self.offsets.get(0, 0.0)
        best, best_T = -np.inf, None
        for T in np.arange(self.t_range[0], self.t_range[1], 0.004):
            exp = np.array([t0 + (c * 12 + k + o - o0) / 12 * T for c in (0, 1) for k, o in self.offsets.items()])
            score = 0.0
            for t, lev in strokes[1:]:
                if t > t0 + self.listen:
                    break
                if np.min(np.abs(exp - t)) < 0.2 * T / 12:
                    score += lev
                else:
                    score -= 0.5 * lev
            if score > best:
                best, best_T = score, T
        self.x = np.array([t0 - o0 / 12 * best_T, best_T])
        self.P = np.diag([0.01 ** 2, (0.03 * best_T) ** 2])
        self.cycle = 0
        self.levels = [l0]
        self.recent = [(t0, l0)]
        for t, lev in strokes[1:]:
            self.observe(t, lev)

    def start(self, t_first: float, t_second: float, k_second: int = 2) -> None:
        """First stroke on pulse 0, the next on pulse ``k_second`` (2 for jembe 2)."""
        o0, o2 = self.offsets.get(0, 0.0), self.offsets.get(k_second, 0.0)
        T = 12 * (t_second - t_first) / (k_second + o2 - o0)
        self.x = np.array([t_first - o0 / 12 * T, T])
        self.P = np.diag([0.02 ** 2, (0.15 * T) ** 2])
        self.cycle = 0

    def pulse_time(self, k: float, cycle: int | None = None) -> float:
        """Predicted time of (fractional) pulse k of a cycle (default: the current one)."""
        c = self.cycle if cycle is None else cycle
        start, T = self.x
        return start + (c - self.cycle) * T + k / 12 * T

    def _roll(self) -> None:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag([0.002 ** 2, self.tempo_drift ** 2])
        self.cycle += 1

    def advance_to(self, t: float) -> None:
        """Roll forward the cycles that have ended before time t."""
        while t > self.x[0] + self.x[1] * (1 + 0.5 / 12):
            self._roll()

    def observe(self, t: float, level: float | None = None) -> tuple[int, int] | None:
        """Assign a stroke at time t to a (cycle, pulse) and correct the state; None if gated out."""
        if level is not None and self.levels and level < self.level_floor * np.median(self.levels[-30:]):
            return None
        if self.own and level is not None and level <= self.own_level:
            i = np.searchsorted(self.own, t)
            if any(abs(self.own[j] - t) < 0.015 for j in (i - 1, i) if 0 <= j < len(self.own)):
                return None
        self.recent = [r for r in self.recent if r[0] > t - self.listen] + [(t, level or 1.0)]
        self.advance_to(t)
        best = None
        for c in (self.cycle, self.cycle + 1):
            for k, o in self.offsets.items():
                e = t - self.pulse_time(k + o, c)
                if best is None or abs(e) < abs(best[2]):
                    best = (c, k, e)
        c, k, e = best
        pulse = self.x[1] / 12
        H = np.array([1.0, (k + self.offsets[k]) / 12 + (c - self.cycle)])
        R = (self.obs_sd * pulse) ** 2
        S = H @ self.P @ H + R
        if abs(e) > min(0.6 * pulse, max(self.gate * pulse, 2.5 * np.sqrt(S))):
            self.misses += 1
            if self.misses >= self.reacquire_after:
                self._reacquire()
            return None
        self.misses = 0
        if c > self.cycle:
            self._roll()
        H = np.array([1.0, (k + self.offsets[k]) / 12])
        S = H @ self.P @ H + R
        K = self.P @ H / S
        self.x = self.x + K * e
        self.P = self.P - np.outer(K, H) @ self.P
        self.log.append((t, self.cycle, k, e))
        self.tempi.append(self.x[1])
        if level is not None:
            self.levels.append(level)
        return self.cycle, k

    def ignore_own(self, times) -> None:
        """Strokes played by the system (sorted times [s]); their echo in the microphone is ignored."""
        self.own = sorted(times)

    def _reacquire(self) -> None:
        t = np.array([r[0] for r in self.recent])
        w = np.array([r[1] for r in self.recent])
        pos = np.array([k + o for k, o in self.offsets.items()])
        start, T_now = self.x
        T0 = float(np.median(self.tempi[-200:])) if self.tempi else T_now
        best = (-np.inf, start, T_now)
        for T in T0 * np.arange(0.85, 1.15, 0.005):
            s = start + np.arange(-T / 2, T / 2, 0.003)
            u = np.mod((t[None, :] - s[:, None]) / T * 12, 12)                # pulse position of each stroke
            d = np.abs(u[:, :, None] - pos[None, None, :])
            d = np.minimum(d, 12 - d).min(axis=2)
            score = np.where(d < 0.2, w, -0.5 * w).sum(axis=1)
            i = int(np.argmax(score))
            if score[i] > best[0]:
                best = (score[i], s[i], T)
        _, s_best, T_best = best
        self.x = np.array([s_best, T_best])
        self.P = np.diag([0.01 ** 2, (0.03 * T_best) ** 2])
        self.misses = 0
        self.n_reacquired += 1


def j2_offsets(events, takes, piece: str) -> dict:
    """Median offset of jembe 2 per pulse (fraction of a pulse, against the annotated grid) over ``takes``."""
    ev = events[(events["instrument"] == "J2") & events["take_id"].isin(takes) & (events["piece"] == piece)]
    ev = ev[ev["dev_sd"].abs() <= 0.5]
    med = ev.groupby("sd")["dev_sd"].median()
    return {k: float(med.get(k + 1, 0.0)) for k in J2_PULSES}
