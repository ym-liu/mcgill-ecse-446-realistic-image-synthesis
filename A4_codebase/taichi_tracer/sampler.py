from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material


# Uniform Sampling Methods
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


# Implement BRDF Sampling Methods
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

        local_dir = tm.normalize(
            tm.vec3([w_x, w_y, w_z])
        )  # local direction in canonical orientation

        # get axis of alignment according to material.Ns (diffuse or specular)
        axis_of_alignment = tm.vec3(0.0)
        if material.Ns == 1:  # if brdf diffuse, aligned about n
            axis_of_alignment = normal
        elif material.Ns > 1:  # if brdf phong, aligned about w_r
            axis_of_alignment = tm.normalize((2 * (tm.dot(normal, w_o)) * normal) - w_o)

        # rotate local direction into world coord sys
        # cos lobe or cos-pow lobe aligned about the axis of alignment
        ortho = ortho_frames(axis_of_alignment)
        world_dir = tm.normalize(ortho @ local_dir)

        # return sampled ray direction
        return world_dir

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
        elif alpha > 1:  # if brdf phong
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

        # get diffuse color and specular coefficient from the material
        alpha = material.Ns  # phong exponent / specular coefficient
        rho = material.Kd  # reflectance (r,g,b) / diffuse color

        # compute the BRDF factor
        brdf_factor = tm.vec3(0.0)
        if alpha == 1:  # if brdf diffuse
            brdf_factor = rho
        elif alpha > 1:  # if brdf phong
            brdf_factor = rho * tm.max(tm.dot(normal, w_i), 0.0)

        return brdf_factor


# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# Microfacet BRDF Methods
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


@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        # Find all of the emissive triangles
        emissive_triangle_ids = self.get_emissive_triangle_indices()
        if len(emissive_triangle_ids) == 0:
            self.has_emissive_triangles = False
        else:
            self.has_emissive_triangles = True
            self.n_emissive_triangles = len(emissive_triangle_ids)
            emissive_triangle_ids = np.array(emissive_triangle_ids, dtype=np.int32)
            self.emissive_triangle_ids = ti.field(
                shape=(emissive_triangle_ids.shape[0]), dtype=ti.i32
            )
            self.emissive_triangle_ids.from_numpy(emissive_triangle_ids)

        # Setup for importance sampling
        if self.has_emissive_triangles:
            # Data Fields
            self.emissive_triangle_areas = ti.field(
                shape=(emissive_triangle_ids.shape[0]), dtype=float
            )
            self.cdf = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.total_emissive_area = ti.field(shape=(), dtype=float)

            # Compute
            self.compute_emissive_triangle_areas()
            self.compute_cdf()

    def get_emissive_triangle_indices(self) -> List[int]:
        # Iterate over each triangle, and check for emissivity
        emissive_triangle_ids = []
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            material_id = self.geometry.triangle_material_ids[triangle_id - 1]
            emissivity = self.material_library.materials[material_id].Ke
            if emissivity.norm() > 0:
                emissive_triangle_ids.append(triangle_id)

        return emissive_triangle_ids

    @ti.kernel
    def compute_emissive_triangle_areas(self):
        for i in range(self.n_emissive_triangles):
            triangle_id = self.emissive_triangle_ids[i]
            vert_ids = (
                self.geometry.triangle_vertex_ids[triangle_id - 1] - 1
            )  # Vertices are indexed from 1
            v0 = self.geometry.vertices[vert_ids[0]]
            v1 = self.geometry.vertices[vert_ids[1]]
            v2 = self.geometry.vertices[vert_ids[2]]

            triangle_area = self.compute_triangle_area(v0, v1, v2)
            self.emissive_triangle_areas[i] = triangle_area
            self.total_emissive_area[None] += triangle_area

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        # Compute Area of a triangle given the 3 vertices
        # Area of a triangle ABC = 0.5 * | AB cross AC |
        return 0.5 * tm.length(tm.cross(v0 - v1, v0 - v2))

    @ti.kernel
    def compute_cdf(self):
        # Compute the CDF of your emissive triangles
        # self.cdf[i] = ...

        # initialize area_sum (cumulative sum of areas)
        area_sum = 0.0

        # compute cumulative sum (raw cdf) of areas
        ti.loop_config(serialize=True)  # serialized loop
        for i in ti.ndrange(self.emissive_triangle_areas.shape[0]):
            area_sum += self.emissive_triangle_areas[i]
            self.cdf[i] = area_sum

        # normalize cdf (by dividing by total area)
        for i in ti.ndrange(self.emissive_triangle_areas.shape[0]):
            self.cdf[i] /= area_sum

    @ti.func
    def sample_emissive_triangle(self) -> int:
        # Sample an emissive triangle using the CDF
        # return the **index** of the triangle

        # generate random variable between [0,1]
        rand_var = ti.random()

        # binary search boundaries
        left = 0
        right = self.emissive_triangle_areas.shape[0] - 1

        # binary search
        # find smallest triangle index with self.cdf[index] >= rand_var
        while left < right:
            mid = (left + right) // 2
            if self.cdf[mid] < rand_var:
                left = mid + 1
            else:
                right = mid

        # return index of sampled triangle
        return left

    @ti.func
    def evaluate_probability(self) -> float:
        # return probability of sampled direction w_i
        # (converted from a sampled point y_i)

        return 1.0 / self.total_emissive_area[None]

    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        sampled_light_triangle_idx = self.sample_emissive_triangle()
        sampled_light_triangle = self.emissive_triangle_ids[sampled_light_triangle_idx]

        # Grab Vertices
        vert_ids = (
            self.geometry.triangle_vertex_ids[sampled_light_triangle - 1] - 1
        )  # Vertices are indexed from 1

        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # generate point on triangle using random barycentric coordinates
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#Sampling
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#SampleUniformTriangle

        # Sample a direction towards mesh light
        # given your sampled triangle vertices
        # generat random barycentric coordinates
        # calculate the light direction
        # light direction = (point on light - hit point)
        # don't forget to normalize!

        # generate 2 canonical random variables with rand_var0 < rand_var1
        rand_var0 = ti.random()
        rand_var1 = ti.random()
        if rand_var0 > rand_var1:  # if rand_var0 > rand_var1, then switch the two
            tmp = rand_var0
            rand_var0 = rand_var1
            rand_var1 = tmp  # rand_var0

        # compute barycentric coordinates b0, b1, b2
        b0 = rand_var0 / 2.0
        b1 = rand_var1 - b0
        b2 = 1.0 - b0 - b1

        # compute sampled surface point y_i
        y_i = (b0 * v0) + (b1 * v1) + (b2 * v2)

        # compute light direction w_i
        light_direction = tm.normalize(y_i - hit_point)

        # return light direction and index of sampled triangle
        return light_direction, sampled_light_triangle


@ti.func
def ortho_frames(axis_of_alignment: tm.vec3) -> tm.mat3:
    # code from assignment 2 tutorial

    random_vec = tm.normalize(tm.vec3([ti.random(), ti.random(), ti.random()]))

    x_axis = tm.cross(axis_of_alignment, random_vec)
    x_axis = tm.normalize(x_axis)

    y_axis = tm.cross(x_axis, axis_of_alignment)
    y_axis = tm.normalize(y_axis)

    ortho_frames = tm.mat3([x_axis, y_axis, axis_of_alignment]).transpose()

    return ortho_frames


@ti.func
def reflect(ray_direction: tm.vec3, normal: tm.vec3) -> tm.vec3:
    pass
