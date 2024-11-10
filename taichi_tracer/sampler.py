from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material


# TODO: Implement Uniform Sampling Methods
@ti.data_oriented
class UniformSampler:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction() -> tm.vec3:
        # generate 2 canonical random variables
        rand_var1 = ti.random()
        rand_var2 = ti.random()

        # generate uniformly-sampled ray direction w_i = (w_x, w_y, w_z)
        w_z = (2 * rand_var1) - 1

        r = tm.sqrt(1 - (w_z * w_z))
        phi = 2 * tm.pi * rand_var2

        w_x = r * tm.cos(phi)
        w_y = r * tm.sin(phi)

        # return sampled ray direction
        return tm.normalize(tm.vec3([w_x, w_y, w_z]))

    @staticmethod
    @ti.func
    def evaluate_probability() -> float:
        return 1.0 / (4.0 * tm.pi)


# TODO: Implement BRDF Sampling Methods
@ti.data_oriented
class BRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        # generate 2 canonical random variables
        rand_var1 = ti.random()
        rand_var2 = ti.random()

        # generate BRDF importance-sampled ray direction w_i = (w_x, w_y, w_z)
        w_z = tm.pow(rand_var1, 1 / (material.Ns + 1))

        r = tm.sqrt(1 - (w_z * w_z))
        phi = 2 * tm.pi * rand_var2

        w_x = r * tm.cos(phi)
        w_y = r * tm.sin(phi)

        local_dir = tm.vec3([w_x, w_y, w_z])  # local direction in canonical orientation

        # rotate local direction into world coord sys at shade point
        tangent = tm.normalize(tm.cross(w_o, normal))
        if tangent.norm() < 1e-7:  # if normal almost vertical, choose another
            tangent = tm.normalize(tm.cross(tm.vec3(1.0, 0.0, 0.0), normal))
        bitangent = tm.cross(normal, tangent)
        world_dir = (
            local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal
        )

        # return sampled ray direction
        return tm.normalize(local_dir)

    @staticmethod
    @ti.func
    def evaluate_probability(
        material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3
    ) -> float:
        # initialize probabiblity
        pdf_brdf = 0.0

        # get specular coefficient from the material
        alpha = material.Ns  # phong exponent / specular coefficient

        # compute the reflected view-direction w_r
        w_r = (2 * (tm.dot(normal, w_o)) * normal) - w_o

        # compute the probability
        if alpha == 1:  # if brdf diffuse
            pdf_brdf = (1 / tm.pi) * tm.max(0, tm.dot(normal, w_i))
        elif alpha > 1:  # if brdf phong
            pdf_brdf = ((alpha + 1) / (2 * tm.pi)) * tm.max(
                0.0, tm.pow(tm.dot(w_r, w_i), alpha)
            )

        # return the probability
        return pdf_brdf

    @staticmethod
    @ti.func
    def evaluate_brdf(
        material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3
    ) -> tm.vec3:
        # compute the reflected view-direction w_r
        w_r = (2 * (tm.dot(normal, w_o)) * normal) - w_o

        # get diffuse color and specular coefficient from the material
        alpha = material.Ns  # phong exponent / specular coefficient
        rho = material.Kd  # reflectance (r,g,b) / diffuse color

        # compute the BRDF
        f_r = tm.vec3(0.0)
        if alpha == 1:  # if brdf diffuse
            f_r = rho / tm.pi
        else:  # if brdf phong
            f_r = ((rho * (alpha + 1)) / (2 * tm.pi)) * tm.max(
                0.0, tm.pow(tm.dot(w_r, w_i), alpha)
            )

        # return the BRDF
        return f_r

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(
        material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3
    ) -> tm.vec3:
        pass


# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# TODO: Implement Microfacet BRDF Methods
# 546 only deliverable
@ti.data_oriented
class MicrofacetBRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass

    @staticmethod
    @ti.func
    def evaluate_probability(
        material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3
    ) -> float:
        pass

    @staticmethod
    @ti.func
    def evaluate_brdf(
        material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3
    ) -> tm.vec3:
        pass


"""
Ignore for now
"""


@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        pass

    def get_emissive_triangle_indices(self) -> List[int]:
        pass

    @ti.kernel
    def compute_emissive_triangle_areas(self):
        pass

    @ti.func
    def compute_triangle_area_given_id(self, triangle_id: int) -> float:
        pass

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        pass

    @ti.kernel
    def compute_cdf(self):
        pass

    @ti.func
    def sample_emissive_triangle(self) -> int:
        pass

    @ti.func
    def evaluate_probability(self) -> float:
        pass

    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        pass


@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    pass


@ti.func
def reflect(ray_direction: tm.vec3, normal: tm.vec3) -> tm.vec3:
    pass
