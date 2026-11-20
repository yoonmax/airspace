"""Procedural meshes and model import for the 3D viewer.

All built-in models are generated in code (original geometry, no assets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) float32 in [0, 1]
    indices: np.ndarray | None  # (M, 3) uint32, or None for unindexed (flat) meshes
    name: str = "mesh"
    has_colors: bool = False

    @property
    def triangle_count(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.positions) // 3

    def normalized(self) -> "Mesh":
        """Centre on the origin and scale to fit the unit sphere ("fit model")."""
        p = self.positions.astype(np.float64)
        lo, hi = p.min(axis=0), p.max(axis=0)
        c = (lo + hi) / 2
        p = p - c
        r = float(np.linalg.norm(p, axis=1).max()) or 1.0
        return Mesh((p / r).astype(np.float32), self.normals, self.colors, self.indices, self.name, self.has_colors)


def _vertex_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = np.zeros_like(pos, dtype=np.float64)
    tri = pos[idx]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, idx[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return (n / ln).astype(np.float32)


def _mesh(pos, idx, name, colors=None) -> Mesh:
    pos = np.asarray(pos, np.float32)
    idx = np.asarray(idx, np.uint32)
    col = np.ones_like(pos) if colors is None else np.asarray(colors, np.float32)
    return Mesh(pos, _vertex_normals(pos, idx), col, idx, name, colors is not None)


def flat_shade(mesh: Mesh) -> Mesh:
    """Unindex a mesh and give each triangle its face normal (faceted look)."""
    assert mesh.indices is not None
    idx = mesh.indices.reshape(-1)
    pos = mesh.positions[idx]
    col = mesh.colors[idx]
    tri = pos.reshape(-1, 3, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    normals = np.repeat(fn, 3, axis=0).astype(np.float32)
    return Mesh(pos.astype(np.float32), normals, col.astype(np.float32), None, mesh.name, mesh.has_colors)


# ------------------------------------------------------------- primitives
def rounded_cube(segments: int = 14, radius: float = 0.18) -> Mesh:
    """A softly bevelled cube: subdivided cube faces projected onto a rounded box."""
    pos, idx = [], []
    n = segments
    lin = np.linspace(-1, 1, n + 1)
    faces = [((0, 1, 2), 1), ((0, 1, 2), -1), ((1, 2, 0), 1), ((1, 2, 0), -1), ((2, 0, 1), 1), ((2, 0, 1), -1)]
    for (a, b, c), sign in faces:
        base = len(pos)
        for i in range(n + 1):
            for j in range(n + 1):
                v = [0.0, 0.0, 0.0]
                v[a], v[b], v[c] = lin[i], lin[j] * sign, sign
                pos.append(v)
        for i in range(n):
            for j in range(n):
                q = base + i * (n + 1) + j
                idx += [[q, q + n + 1, q + 1], [q + 1, q + n + 1, q + n + 2]]
    p = np.array(pos, np.float64)
    inner = np.clip(p, -(1 - radius), 1 - radius)
    d = p - inner
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    ln[ln == 0] = 1
    p = inner + d / ln * radius
    m = _mesh(p * 0.82, idx, "cube")
    # Fix winding so normals face outward.
    if np.mean(np.sum(m.normals * m.positions, axis=1)) < 0:
        m = _mesh(m.positions, m.indices[:, ::-1], "cube")
    return m


def uv_sphere(rings: int = 64, segments: int = 96) -> Mesh:
    pos, idx = [], []
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segments + 1):
            ph = 2 * math.pi * j / segments
            pos.append((math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)))
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            idx += [[a, a + 1, b], [a + 1, b + 1, b]]
    m = _mesh(pos, idx, "sphere")
    m.normals = m.positions.copy()
    return m


def torus(R: float = 0.7, r: float = 0.28, seg: int = 128, side: int = 48) -> Mesh:
    pos, nrm, idx = [], [], []
    for i in range(seg + 1):
        u = 2 * math.pi * i / seg
        cu, su = math.cos(u), math.sin(u)
        for j in range(side + 1):
            v = 2 * math.pi * j / side
            cv, sv = math.cos(v), math.sin(v)
            pos.append(((R + r * cv) * cu, r * sv, (R + r * cv) * su))
            nrm.append((cv * cu, sv, cv * su))
    for i in range(seg):
        for j in range(side):
            a = i * (side + 1) + j
            b = a + side + 1
            idx += [[a, b, a + 1], [a + 1, b, b + 1]]
    m = _mesh(pos, idx, "torus")
    m.normals = np.array(nrm, np.float32)
    return m


def torus_knot(p: int = 2, q: int = 3, seg: int = 360, side: int = 28, tube: float = 0.16) -> Mesh:
    def curve(t: float) -> np.ndarray:
        r = 0.55 + 0.25 * math.cos(q * t)
        return np.array([r * math.cos(p * t), 0.25 * math.sin(q * t) * 1.4, r * math.sin(p * t)])

    pos, idx = [], []
    for i in range(seg + 1):
        t = 2 * math.pi * i / seg
        c = curve(t)
        tangent = curve(t + 1e-3) - curve(t - 1e-3)
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(normal) < 1e-6:
            normal = np.array([1.0, 0.0, 0.0])
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        for j in range(side + 1):
            v = 2 * math.pi * j / side
            pos.append(c + tube * (math.cos(v) * normal + math.sin(v) * binormal))
    for i in range(seg):
        for j in range(side):
            a = i * (side + 1) + j
            b = a + side + 1
            idx += [[a, b, a + 1], [a + 1, b, b + 1]]
    m = _mesh(pos, idx, "knot")
    if np.mean(np.sum(m.normals * (m.positions - m.positions.mean(0)), axis=1)) < 0:
        m = _mesh(m.positions, m.indices[:, ::-1], "knot")
    return m


def icosphere(subdiv: int = 3) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(x, np.float64) / np.linalg.norm(x) for x in verts]
    cache: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = v[a] + v[b]
            v.append(m / np.linalg.norm(m))
            cache[key] = len(v) - 1
        return cache[key]

    for _ in range(subdiv):
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    return np.array(v, np.float32), np.array(faces, np.uint32)


def _value_noise3(p: np.ndarray, seed: int = 0, octaves: int = 5) -> np.ndarray:
    """Smooth fractal noise from layered random sinusoids (deterministic, dependency-free)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(p))
    amp, freq = 1.0, 1.4
    for _ in range(octaves):
        for _ in range(3):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            phase = rng.uniform(0, 2 * np.pi)
            out += amp * np.sin(p @ d * freq * 2.2 + phase)
        amp *= 0.55
        freq *= 2.03
    return out / np.abs(out).max()


def sculpture() -> Mesh:
    """Low-poly faceted stone: a displaced icosphere, flat shaded."""
    v, f = icosphere(2)
    n = _value_noise3(v.astype(np.float64), seed=7, octaves=3)
    disp = 1 + 0.22 * n
    stretch = np.array([0.85, 1.15, 0.9])
    pos = v * disp[:, None] * stretch
    return flat_shade(_mesh(pos, f, "sculpture"))


def globe() -> Mesh:
    """A procedural planet: noise continents, coloured oceans, polar caps."""
    v, f = icosphere(5)
    h = _value_noise3(v.astype(np.float64), seed=3, octaves=6)
    land = h > 0.08
    lat = np.abs(v[:, 1])
    ocean = np.array([0.10, 0.24, 0.42]) * (0.75 + 0.25 * (h[:, None] + 1) / 2)
    deep = np.array([0.06, 0.14, 0.30])
    ocean = np.where((h < -0.25)[:, None], deep, ocean)
    grass = np.array([0.30, 0.47, 0.30])
    desert = np.array([0.72, 0.62, 0.42])
    mountain = np.array([0.55, 0.52, 0.48])
    t = np.clip((h - 0.08) / 0.6, 0, 1)[:, None]
    landc = np.where((lat < 0.35)[:, None], desert * (1 - t) + grass * t, grass * (1 - t) + mountain * t)
    col = np.where(land[:, None], landc, ocean)
    ice = lat > 0.92 - 0.04 * h
    col = np.where(ice[:, None], np.array([0.92, 0.94, 0.97]), col)
    pos = v * (1 + np.where(land, 0.012 * np.clip(h, 0, 1), 0.0))[:, None]
    m = _mesh(pos, f, "globe", colors=col)
    return m


BUILTIN = {
    "knot": torus_knot,
    "cube": rounded_cube,
    "sphere": uv_sphere,
    "torus": torus,
    "sculpture": sculpture,
    "globe": globe,
}


def load_model_file(path: str) -> Mesh:
    """Load GLB/GLTF/OBJ/STL/PLY via trimesh (merged into one mesh)."""
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("No triangle geometry found in this file")
    if len(loaded.faces) > 2_000_000:
        raise ValueError("Model is too large (over 2M triangles)")
    pos = np.asarray(loaded.vertices, np.float32)
    idx = np.asarray(loaded.faces, np.uint32)
    normals = np.asarray(loaded.vertex_normals, np.float32)
    colors = None
    try:
        vc = loaded.visual.to_color().vertex_colors  # type: ignore[union-attr]
        if vc is not None and len(vc) == len(pos):
            colors = np.asarray(vc[:, :3], np.float32) / 255.0
            if np.allclose(colors, colors[0]):
                colors = None
    except Exception:  # noqa: BLE001 - colours are optional
        colors = None
    mesh = Mesh(pos, normals, np.ones_like(pos) if colors is None else colors, idx,
                name=path.rsplit("/", 1)[-1], has_colors=colors is not None)
    return mesh.normalized()
