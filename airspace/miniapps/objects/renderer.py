"""Offscreen ModernGL renderer for the objects viewer.

Renders into a multisampled framebuffer, resolves, and reads back RGBA for
compositing into the glass window. All GL calls happen inside
``with self.ctx:`` so the standalone context never disturbs Qt's own
OpenGL context.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from airspace.miniapps.objects.models import Mesh

log = logging.getLogger(__name__)

VERT = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_world;
out vec3 v_normal;
out vec3 v_color;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = normalize(u_normal * in_normal);
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 330
uniform vec3 u_camera;
uniform vec3 u_base;
uniform float u_rough;
uniform float u_metal;
uniform float u_fresnel;
uniform float u_vertex_color;
uniform float u_alpha;
uniform vec3 u_key_dir;  uniform vec3 u_key_col;
uniform vec3 u_fill_dir; uniform vec3 u_fill_col;
uniform vec3 u_rim_dir;  uniform vec3 u_rim_col;
uniform vec3 u_sky; uniform vec3 u_ground;
uniform vec3 u_wire_col;
uniform float u_wire;
in vec3 v_world;
in vec3 v_normal;
in vec3 v_color;
out vec4 f_color;

vec3 light(vec3 n, vec3 v, vec3 l, vec3 col, vec3 albedo) {
    l = normalize(l);
    float ndl = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float shininess = mix(220.0, 6.0, u_rough);
    float spec = pow(max(dot(n, h), 0.0), shininess) * (shininess + 8.0) / 25.0;
    vec3 f0 = mix(vec3(0.04), albedo, u_metal);
    vec3 diffuse = albedo * (1.0 - u_metal) * ndl;
    return col * (diffuse + f0 * spec * ndl);
}

void main() {
    if (u_wire > 0.5) { f_color = vec4(u_wire_col * 0.85, 0.85); return; }
    vec3 n = normalize(v_normal);
    vec3 v = normalize(u_camera - v_world);
    if (dot(n, v) < 0.0) n = -n;
    vec3 albedo = mix(u_base, v_color * (0.6 + 0.4 * u_base), u_vertex_color);
    vec3 ambient = mix(u_ground, u_sky, n.y * 0.5 + 0.5) * albedo * (1.0 - 0.6 * u_metal);
    vec3 c = ambient;
    c += light(n, v, u_key_dir, u_key_col, albedo);
    c += light(n, v, u_fill_dir, u_fill_col, albedo);
    float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    // Environment reflection from a sky/ground gradient (cheap stand-in for an HDRI).
    vec3 r = reflect(-v, n);
    vec3 env = mix(u_ground * 1.5, u_sky * 2.4, smoothstep(-0.25, 0.65, r.y));
    vec3 f0 = mix(vec3(0.04), albedo, u_metal);
    c += env * (f0 + (1.0 - f0) * fres) * (1.0 - 0.75 * u_rough);
    c += u_rim_col * fres * u_fresnel * 0.5;
    c += light(n, v, u_rim_dir, u_rim_col, albedo) * 0.6;
    // ACES-ish tone map and gamma.
    c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    float a = mix(u_alpha, 1.0, fres * (1.0 - u_alpha));
    f_color = vec4(c * a, a);
}
"""

SHADOW_VERT = """
#version 330
uniform mat4 u_vp;
uniform float u_y;
uniform float u_size;
in vec2 in_xy;
out vec2 v_uv;
void main() {
    v_uv = in_xy;
    gl_Position = u_vp * vec4(in_xy.x * u_size, u_y, in_xy.y * u_size, 1.0);
}
"""

SHADOW_FRAG = """
#version 330
uniform float u_strength;
in vec2 v_uv;
out vec4 f_color;
void main() {
    float d = length(v_uv);
    float a = smoothstep(1.0, 0.0, d);
    a = a * a * u_strength;
    f_color = vec4(0.0, 0.0, 0.0, a);
}
"""


@dataclass(frozen=True)
class Material:
    name: str
    base: tuple[float, float, float]
    rough: float
    metal: float
    fresnel: float
    alpha: float = 1.0
    vertex_color: float = 0.0


MATERIALS = [
    Material("clay", (0.78, 0.70, 0.63), 0.85, 0.0, 0.35),
    Material("ceramic", (0.80, 0.83, 0.88), 0.22, 0.0, 0.55),
    Material("metal", (0.95, 0.80, 0.55), 0.28, 1.0, 0.25),
    Material("glass", (0.30, 0.42, 0.55), 0.08, 0.0, 1.2, alpha=0.38),
    Material("natural", (1.0, 1.0, 1.0), 0.6, 0.0, 0.4, vertex_color=1.0),
]

LIGHTING = {
    "studio": dict(key=((0.6, 0.9, 0.7), (1.45, 1.40, 1.32)), fill=((-0.8, 0.3, 0.4), (0.32, 0.36, 0.44)),
                   rim=((-0.3, 0.5, -1.0), (0.7, 0.75, 0.8)), sky=(0.26, 0.28, 0.32), ground=(0.07, 0.065, 0.065)),
    "warm": dict(key=((0.7, 0.6, 0.6), (1.7, 1.25, 0.9)), fill=((-0.8, 0.2, 0.3), (0.25, 0.22, 0.34)),
                 rim=((0.0, 0.4, -1.0), (1.2, 0.8, 0.5)), sky=(0.30, 0.25, 0.22), ground=(0.10, 0.07, 0.06)),
    "cool": dict(key=((-0.5, 0.8, 0.6), (1.05, 1.25, 1.6)), fill=((0.9, 0.1, 0.3), (0.22, 0.30, 0.40)),
                 rim=((0.2, 0.3, -1.0), (0.6, 0.9, 1.3)), sky=(0.22, 0.28, 0.36), ground=(0.06, 0.07, 0.09)),
    "dramatic": dict(key=((0.9, 0.6, 0.1), (2.2, 2.0, 1.8)), fill=((-0.9, 0.0, 0.2), (0.06, 0.06, 0.08)),
                     rim=((-0.5, 0.2, -1.0), (1.6, 1.6, 1.8)), sky=(0.08, 0.08, 0.10), ground=(0.02, 0.02, 0.02)),
}


# ------------------------------------------------------------- matrices
def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], np.float64)


def orthonormalize(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    return u @ vt


# --------------------------------------------------------------- renderer
class ObjectRenderer:
    def __init__(self) -> None:
        import moderngl

        self.ctx = moderngl.create_standalone_context(require=330)
        self.samples = min(4, self.ctx.max_samples)
        with self.ctx:
            self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
            self.shadow_prog = self.ctx.program(vertex_shader=SHADOW_VERT, fragment_shader=SHADOW_FRAG)
            quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
            self.shadow_vbo = self.ctx.buffer(quad.tobytes())
            self.shadow_vao = self.ctx.vertex_array(self.shadow_prog, [(self.shadow_vbo, "2f", "in_xy")])
        self.vao = None
        self._buffers: list = []
        self.size = (0, 0)
        self.fbo = self.resolve = None
        self.mesh: Mesh | None = None
        self.min_y = -1.0

    def set_mesh(self, mesh: Mesh) -> None:
        with self.ctx:
            for b in self._buffers:
                b.release()
            if self.vao is not None:
                self.vao.release()
            data = np.hstack([mesh.positions, mesh.normals, mesh.colors]).astype(np.float32)
            vbo = self.ctx.buffer(data.tobytes())
            self._buffers = [vbo]
            if mesh.indices is not None:
                ibo = self.ctx.buffer(mesh.indices.astype(np.uint32).tobytes())
                self._buffers.append(ibo)
                self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
                                                 index_buffer=ibo, index_element_size=4)
            else:
                self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")])
        self.mesh = mesh

    def _ensure_fbo(self, w: int, h: int) -> None:
        if (w, h) == self.size and self.fbo is not None:
            return
        for obj in (self.fbo, self.resolve):
            if obj is not None:
                obj.release()
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((w, h), samples=self.samples)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h), samples=self.samples))
        self.resolve = self.ctx.simple_framebuffer((w, h), components=4)
        self.size = (w, h)

    def render(self, w: int, h: int, rot: np.ndarray, distance: float, pan: tuple[float, float], scale: float,
               material: Material, lighting: str, wireframe: bool, accent: tuple[float, float, float]) -> np.ndarray:
        """Render and return premultiplied RGBA (h, w, 4) uint8, top row first."""
        import moderngl

        w, h = max(8, int(w)), max(8, int(h))
        with self.ctx:
            self._ensure_fbo(w, h)
            self.fbo.use()
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA

            proj = perspective(32.0, w / h, 0.05, 50.0)
            eye = np.array([0.0, 0.35, distance], np.float64)
            view = look_at(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
            view[0, 3] += pan[0]
            view[1, 3] += pan[1]
            model = np.eye(4, dtype=np.float32)
            model[:3, :3] = rot * scale
            mvp = proj @ view @ model
            normal = np.linalg.inv(rot.T).astype(np.float32)

            # Soft contact shadow under the object (drawn first, no depth write).
            if self.mesh is not None:
                world_y = (self.mesh.positions @ (rot * scale).T)[:, 1]
                floor = float(world_y.min()) - 0.04
                self.ctx.depth_mask = False
                sp = self.shadow_prog
                sp["u_vp"].write((proj @ view).T.astype(np.float32).tobytes())
                sp["u_y"].value = floor
                sp["u_size"].value = 1.15 * scale
                sp["u_strength"].value = 0.55
                self.shadow_vao.render(moderngl.TRIANGLE_STRIP)
                self.ctx.depth_mask = True

            if self.vao is not None:
                p = self.prog
                p["u_mvp"].write(mvp.T.astype(np.float32).tobytes())
                p["u_model"].write(model.T.astype(np.float32).tobytes())
                p["u_normal"].write(normal.T.astype(np.float32).tobytes())
                p["u_camera"].value = tuple(float(x) for x in (np.linalg.inv(view) @ np.array([0, 0, 0, 1.0]))[:3])
                p["u_base"].value = material.base
                p["u_rough"].value = material.rough
                p["u_metal"].value = material.metal
                p["u_fresnel"].value = material.fresnel
                p["u_alpha"].value = material.alpha
                p["u_vertex_color"].value = material.vertex_color if self.mesh and self.mesh.has_colors else 0.0
                L = LIGHTING.get(lighting, LIGHTING["studio"])
                for key in ("key", "fill", "rim"):
                    d, c = L[key]
                    p[f"u_{key}_dir"].value = d
                    p[f"u_{key}_col"].value = c
                p["u_sky"].value = L["sky"]
                p["u_ground"].value = L["ground"]
                p["u_wire_col"].value = accent
                p["u_wire"].value = 0.0
                if material.alpha < 1.0:
                    # Two-pass transparency: back faces then front faces.
                    self.ctx.enable(moderngl.CULL_FACE)
                    self.ctx.front_face = "cw"
                    self.vao.render()
                    self.ctx.front_face = "ccw"
                    self.vao.render()
                    self.ctx.disable(moderngl.CULL_FACE)
                else:
                    self.vao.render()
                if wireframe:
                    self.ctx.wireframe = True
                    self.ctx.depth_func = "<="
                    p["u_wire"].value = 1.0
                    self.ctx.polygon_offset = -1.0, -1.0
                    self.vao.render()
                    self.ctx.wireframe = False
                    self.ctx.depth_func = "<"
                    self.ctx.polygon_offset = 0.0, 0.0
            self.ctx.copy_framebuffer(self.resolve, self.fbo)
            data = self.resolve.read(components=4, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(img[::-1])

    def release(self) -> None:
        try:
            with self.ctx:
                for obj in (*self._buffers, self.vao, self.fbo, self.resolve, self.shadow_vao, self.shadow_vbo,
                            self.prog, self.shadow_prog):
                    if obj is not None:
                        obj.release()
            self.ctx.release()
        except Exception:  # noqa: BLE001
            log.debug("renderer release failed", exc_info=True)
