from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF, MicrofacetBRDF
from .materials import Material


@ti.data_oriented
class A1Renderer:

    # Enumerate the different shading modes
    class ShadeMode(IntEnum):
        HIT = 1
        TRIANGLE_ID = 2
        DISTANCE = 3
        BARYCENTRIC = 4
        NORMAL = 5
        MATERIAL_ID = 6

    def __init__(self, width: int, height: int, scene_data: SceneData) -> None:

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.scene_data = scene_data

        self.shade_mode = ti.field(shape=(), dtype=int)
        self.set_shade_hit()

        # Distance at which the distance shader saturates
        self.max_distance = 10.0

        # Numbers used to generate colors for integer index values
        self.r = 3.14159265
        self.b = 2.71828182
        self.g = 6.62607015

        # initialize iteration counter to 0
        self.iter_counter = ti.field(dtype=float, shape=())
        self.iter_counter[None] = 0

    def set_shade_hit(self):
        self.shade_mode[None] = self.ShadeMode.HIT

    def set_shade_triangle_ID(self):
        self.shade_mode[None] = self.ShadeMode.TRIANGLE_ID

    def set_shade_distance(self):
        self.shade_mode[None] = self.ShadeMode.DISTANCE

    def set_shade_barycentrics(self):
        self.shade_mode[None] = self.ShadeMode.BARYCENTRIC

    def set_shade_normal(self):
        self.shade_mode[None] = self.ShadeMode.NORMAL

    def set_shade_material_ID(self):
        self.shade_mode[None] = self.ShadeMode.MATERIAL_ID

    @ti.kernel
    def render(self):
        # increment iteration counter
        self.iter_counter[None] += 1

        # for all pixels in the image plane
        for x, y in ti.ndrange(self.width, self.height):
            # TODO: Change the naive renderer to do progressive rendering
            """
            - call generate_ray with jitter = True
            - progressively accumulate the pixel values in each canvas [x, y] position
            """
            # generate and shade ray in a random location within the pixel
            random_ray = self.camera.generate_ray(x, y, True)

            # progressively accumulate the pixel values in each canvas [x, y] position
            color = self.shade_ray(random_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / tm.vec3(
                self.iter_counter[None]
            )

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        color = tm.vec3(0)
        if self.shade_mode[None] == int(self.ShadeMode.HIT):
            color = self.shade_hit(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.TRIANGLE_ID):
            color = self.shade_triangle_id(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.DISTANCE):
            color = self.shade_distance(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.BARYCENTRIC):
            color = self.shade_barycentric(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.NORMAL):
            color = self.shade_normal(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.MATERIAL_ID):
            color = self.shade_material_id(hit_data)
        return color

    @ti.func
    def shade_hit(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            if not hit_data.is_backfacing:
                color = tm.vec3(1)
            else:
                color = tm.vec3([0.5, 0, 0])
        return color

    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1  # Add 1 so that ID 0 is not black
            r = triangle_id * self.r % 1
            g = triangle_id * self.g % 1
            b = triangle_id * self.b % 1
            color = tm.vec3(r, g, b)
        return color

    @ti.func
    def shade_distance(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            d = tm.clamp(hit_data.distance / self.max_distance, 0, 1)
            color = tm.vec3(d)
        return color

    @ti.func
    def shade_barycentric(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            u = hit_data.barycentric_coords[0]
            v = hit_data.barycentric_coords[1]
            w = 1.0 - u - v
            color = tm.vec3(u, v, w)
        return color

    @ti.func
    def shade_normal(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            normal = hit_data.normal
            color = (normal + 1.0) / 2.0  # Scale to range [0,1]
        return color

    @ti.func
    def shade_material_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            material_id = hit_data.material_id + 1  # Add 1 so that ID 0 is not black
            r = material_id * self.r % 1
            g = material_id * self.g % 1
            b = material_id * self.b % 1
            color = tm.vec3(r, g, b)
        return color


@ti.data_oriented
class A2Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__(self, width: int, height: int, scene_data: SceneData) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()

    def set_sample_uniform(self):
        self.sample_mode[None] = self.SampleMode.UNIFORM

    def set_sample_brdf(self):
        self.sample_mode[None] = self.SampleMode.BRDF

    def set_sample_microfacet(self):
        self.sample_mode[None] = self.SampleMode.MICROFACET

    @ti.kernel
    def render(self):
        # increment iteration counter
        self.iter_counter[None] += 1

        # for all pixels in the image plane
        for x, y in ti.ndrange(self.width, self.height):
            # TODO: Change the naive renderer to do progressive rendering
            """
            - call generate_ray with jitter = True
            - progressively accumulate the pixel values in each canvas [x, y] position
            """
            # generate and shade ray in a random location within the pixel
            random_ray = self.camera.generate_ray(x, y, True)

            # progressively accumulate the pixel values in each canvas [x, y] position
            color = self.shade_ray(random_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / tm.vec3(
                self.iter_counter[None]
            )

    def reset(self):
        self.canvas.fill(0.0)
        self.iter_counter.fill(0.0)

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:

        # initialize color vector
        color = tm.vec3(0.0)

        # get hit data from the ray
        hit_data = self.scene_data.ray_intersector.query_ray(ray)

        # get surface-ray intersection from hit_data
        x = tm.vec3(0.0)  # surface-ray intersection
        if hit_data.is_hit:
            x = ray.origin + (hit_data.distance * ray.direction)

        # get material from hit_data
        material = self.scene_data.material_library.materials[hit_data.material_id]

        # if our ray hits an object
        if hit_data.is_hit:

            # normal: get object surface normal from hit_data
            normal = hit_data.normal

            # w_o: compute direction opposite of eye ray
            w_o = -ray.direction

            # w_i: sample direction
            w_i = tm.vec3(0.0)
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                # generate uniformly-sampled ray direction
                w_i = UniformSampler.sample_direction()
            elif self.sample_mode[None] == int(self.SampleMode.BRDF):
                # generate brdf importance-sampled ray direction
                w_i = BRDF.sample_direction(material, w_o, normal)
            elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
                pass

            # L_e: query environment light
            L_e = tm.vec3(0.0)
            reflected_ray = Ray(origin=x, direction=w_i)
            L_e = self.scene_data.environment.query_ray(reflected_ray)

            # V: perform occlusion check
            V = 1  # visibility function
            shadow_ray = Ray()  # construct shadow ray from the surface to the light
            shadow_ray.origin = x + (normal * self.RAY_OFFSET)  # surface point
            shadow_ray.direction = w_i  # direction from surface to light
            shadow_ray_hit_data = self.scene_data.ray_intersector.query_ray(shadow_ray)
            if shadow_ray_hit_data.is_hit:  # if hit, then occluded
                V = 0

            # uniform importance sampling
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                # brdf: compute the BRDF
                brdf = BRDF.evaluate_brdf(material, w_o, w_i, normal)

                # pdf: evaluate probability
                pdf = UniformSampler.evaluate_probability()

                # compute color using monte carlo integration
                color += (L_e * V * brdf * tm.max(tm.dot(normal, w_i), 0.0)) / pdf

            # brdf importance sampling
            elif self.sample_mode[None] == int(self.SampleMode.BRDF):
                # brdf_factor: compute the BRDF factor
                brdf_factor = BRDF.evaluate_brdf_factor(material, w_o, w_i, normal)

                # compute color using monte carlo integration
                color += L_e * V * brdf_factor

            # microfacet brdf importance sampling
            elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
                pass

        # if our ray doesnt hit an object then it's the environment
        else:
            color = self.scene_data.environment.query_ray(ray)

        return color
