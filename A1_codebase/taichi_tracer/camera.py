import taichi as ti
import taichi.math as tm
import numpy as np

from .ray import Ray


@ti.data_oriented
class Camera:

    def __init__(self, width: int = 128, height: int = 128) -> None:

        # Camera pixel width and height are fixed
        self.width = width
        self.height = height

        # Camera parameters that can be modified are stored as fields
        self.eye = ti.Vector.field(n=3, shape=(), dtype=float)
        self.at = ti.Vector.field(n=3, shape=(), dtype=float)
        self.up = ti.Vector.field(n=3, shape=(), dtype=float)
        self.fov = ti.field(shape=(), dtype=float)

        self.x = ti.Vector.field(n=3, shape=(), dtype=float)
        self.y = ti.Vector.field(n=3, shape=(), dtype=float)
        self.z = ti.Vector.field(n=3, shape=(), dtype=float)

        self.camera_to_world = ti.Matrix.field(n=4, m=4, shape=(), dtype=float)

        # Initialize with some default params
        self.set_camera_parameters(
            eye=tm.vec3([0, 0, 5]),
            at=tm.vec3([0, 0, 0]),
            up=tm.vec3([0, 1, 0]),
            fov=60.0,
        )

    def set_camera_parameters(
        self,
        eye: tm.vec3 = None,
        at: tm.vec3 = None,
        up: tm.vec3 = None,
        fov: float = None,
    ) -> None:

        if eye:
            self.eye[None] = eye
        if at:
            self.at[None] = at
        if up:
            self.up[None] = up
        if fov:
            self.fov[None] = fov
        self.compute_matrix()

    @ti.kernel
    def compute_matrix(self):
        """
        TODO: Compute Camera to World Matrix

        self.camera_to_world[None] = tm.mat4(<Your Matrix>)
        """

        # z_c = normalized vector from the eye to look-at point
        z = tm.normalize(self.at[None] - self.eye[None])

        # x_c = up_w cross z_c
        x = tm.cross(self.up[None], z)

        # y_c = z_c cross x_c
        y = tm.cross(z, x)

        # update self x,y,z
        self.x[None], self.y[None], self.z[None] = x, y, z

        # build matrix
        self.camera_to_world[None] = tm.mat4(
            [x.x, y.x, z.x, self.eye[None].x],
            [x.y, y.y, z.y, self.eye[None].y],
            [x.z, y.z, z.z, self.eye[None].z],
            [0, 0, 0, 1],
        )

    @ti.func
    def generate_ray(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> Ray:
        """
        TODO: Generate Ray

        - generate ndc coords
        - generate camera coods from NDC coords
        - generate a ray
            - ray = Ray()
        - set the ray direction and origin
            - ray.origin = ...
            - ray.direction = ...
        - Ignore jittering for now
        - return ray
        """

        # generate ndc coords
        ndc_coords = Camera.generate_ndc_coords(self, pixel_x, pixel_y, jitter)

        # generate camera coords from NDC coords
        cam_coords = Camera.generate_camera_coords(self, ndc_coords)

        # generate a ray
        ray = Ray()

        # set the ray direction and origin
        ray.origin = self.eye[None]
        direction = self.camera_to_world[None] @ cam_coords
        ray.direction = tm.normalize(tm.vec3([direction.x, direction.y, direction.z]))

        # return ray
        return ray

    @ti.func
    def generate_ndc_coords(
        self, pixel_x: int, pixel_y: int, jitter: bool = False
    ) -> tm.vec2:
        """
        TODO: Generate NDC coords

        - Given screen coords pixel_x and pixel_y,
          calculate ndc coords ndc_x and ndc_y

        - Ignore jittering for now

        return tm.vec2([ndc_x, ndc_y])
        """

        # calculate ndc coords ndc_x and ndc_y
        # first normalize screen coordinates, then adjust to ndc
        ndc_x = ((float(pixel_x + 0.5) / self.width) * 2) - 1
        ndc_y = ((float(pixel_y + 0.5) / self.height) * 2) - 1

        return tm.vec2([ndc_x, ndc_y])

    @ti.func
    def generate_camera_coords(self, ndc_coords: tm.vec2) -> tm.vec4:
        """
        TODO: Generate Camera coordinates
        - compute camera_x, camera_y, camera_z
        - return tm.vec4([camera_x, camera_y, camera_z, 0.0])
        """

        # cam_z: given valid ndc_coords, the z_c is always 1
        cam_z = 1.0

        # cam_y: tan(vertical fov / 2) = opp / cam_z
        # where opp = cam_y corresponding to 1 unit in ndc
        ndc_to_cam_y = tm.tan(tm.radians(self.fov[None]) / 2) * cam_z
        cam_y = ndc_coords.y * ndc_to_cam_y

        # cam_x: (ndc_to_cam_x / ndc_to_cam_y) = (self.width / self.height)
        ndc_to_cam_x = ndc_to_cam_y * (self.width / self.height)
        cam_x = ndc_coords.x * ndc_to_cam_x

        return tm.vec4([cam_x, cam_y, cam_z, 0.0])
