/****************************************************************************
 * Copyright ©2017 Brian Curless.  All rights reserved.  Permission is hereby
 * granted to students registered for University of Washington CSE 457 or CSE
 * 557 for use solely during Autumn Quarter 2017 for purposes of the course.
 * No other use, copying, distribution, or modification is permitted without
 * prior written consent. Copyrights for third-party components of this work
 * must be honored.  Instructors interested in reusing these course materials
 * should contact the author.
 ****************************************************************************/
#include "trace/raytracer.h"
#include <scene/scene.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <glm/gtx/component_wise.hpp>
#include <scene/components/triangleface.h>
#include <glm/gtx/string_cast.hpp>
#include "components.h"

unsigned int pow4(unsigned int e) {
    unsigned int i = 1;
    while(e>0) {
        i *= 4;
        e--;
    }
    return i;
}

RayTracer::RayTracer(Scene& scene, SceneObject& camobj) :
    trace_scene(&scene, camobj.GetComponent<Camera>()->TraceEnableAcceleration.Get()), next_render_index(0), cancelling(false), first_pass_buffer(nullptr)
{
    Camera* cam = camobj.GetComponent<Camera>();
    settings.skybox = cam->Skybox.Get();

    settings.width = cam->RenderWidth.Get();
    settings.height = cam->RenderHeight.Get();
    settings.pixel_size_x = 1.0/double(settings.width);
    settings.pixel_size_y = 1.0/double(settings.height);

    settings.shadows = cam->TraceShadows.Get() != Camera::TRACESHADOWS_NONE;
    settings.translucent_shadows = cam->TraceShadows.Get() == Camera::TRACESHADOWS_COLORED;
    settings.reflections = cam->TraceEnableReflection.Get();
    settings.refractions = cam->TraceEnableRefraction.Get();

    settings.random_mode = cam->TraceRandomMode.Get();
    settings.diffuse_reflection = cam->TraceEnableReflection.Get() && cam->TraceDiffuseReflection.Get() && settings.random_mode != Camera::TRACERANDOM_DETERMINISTIC;
    settings.caustics = settings.diffuse_reflection && settings.shadows && cam->TraceCaustics.Get();
    settings.random_branching = cam->TraceRandomBranching.Get() && settings.random_mode != Camera::TRACERANDOM_DETERMINISTIC;

    settings.samplecount_mode = cam->TraceSampleCountMode.Get();

    settings.constant_samples_per_pixel = pow4(cam->TraceConstantSampleCount.Get());
    settings.dynamic_sampling_min_depth = cam->TraceSampleMinCount.Get();
    settings.dynamic_sampling_max_depth = cam->TraceSampleMaxCount.Get()+1;
    settings.adaptive_max_diff_squared = cam->TraceAdaptiveSamplingMaxDiff.Get();
    settings.adaptive_max_diff_squared *= settings.adaptive_max_diff_squared;
    settings.max_stderr = cam->TraceStdErrorSamplingCutoff.Get();

    if (settings.samplecount_mode == Camera::TRACESAMPLING_RECURSIVE && settings.random_mode!=Camera::TRACERANDOM_DETERMINISTIC) {
        qDebug() << "Adaptive Recursive Supersampling does not work with Monte Carlo!";
        settings.samplecount_mode = Camera::TRACESAMPLING_CONSTANT;
    }

    settings.max_depth = cam->TraceMaxDepth.Get();

    //camera looks -z, x is right, y is up
    glm::mat4 camera_matrix = camobj.GetModelMatrix();

    settings.projection_origin = glm::vec3(camera_matrix * glm::vec4(0,0,0,1));

    glm::dvec3 fw_point = glm::dvec3(camera_matrix * glm::vec4(0,0,-1,1));
    glm::dvec3 x_point = glm::dvec3(camera_matrix * glm::vec4(1,0,0,1));
    glm::dvec3 y_point = glm::dvec3(camera_matrix * glm::vec4(0,1,0,1));
    glm::dvec3 fw_vec = glm::normalize(fw_point - settings.projection_origin);
    glm::dvec3 x_vec = glm::normalize(x_point - settings.projection_origin);
    glm::dvec3 y_vec = glm::normalize(y_point - settings.projection_origin);

    //FOV is full vertical FOV
    double tangent = tan( (cam->FOV.Get()/2.0) * 3.14159/180.0 );

    double focus_dist = cam->TraceFocusDistance.Get();

    settings.projection_forward = focus_dist * fw_vec;
    settings.projection_up = focus_dist * tangent * y_vec;
    settings.projection_right = focus_dist * tangent * AspectRatio() * x_vec;

    double aperture_radius = cam->TraceApertureSize.Get() * 0.5;
    settings.aperture_up = aperture_radius * y_vec;
    settings.aperture_right = aperture_radius * x_vec;
    settings.aperture_radius = aperture_radius;

    buffer = new uint8_t[settings.width * settings.height * 3]();

    int num_threads = QThread::idealThreadCount();
    if (num_threads > 1) {
        num_threads -= 1; //leave a free thread so the computer doesn't totally die
    }
    thread_pool.setMaxThreadCount(num_threads);

    if (settings.samplecount_mode!=Camera::TRACESAMPLING_CONSTANT && settings.dynamic_sampling_min_depth==0) {
        int orig = settings.samplecount_mode;
        settings.samplecount_mode = Camera::TRACESAMPLING_CONSTANT;
        settings.constant_samples_per_pixel = 1;

        //TODO make it not hang here
        for (unsigned int i = 0; i < num_threads; i++) {
            thread_pool.start(new RTWorker(*this));
        }
        thread_pool.waitForDone(-1);
        next_render_index.store(0);

        settings.samplecount_mode = orig;
        first_pass_buffer = buffer;
        buffer = new uint8_t[settings.width * settings.height * 3]();
    }

    // Spin off threads
    for (unsigned int i = 0; i < num_threads; i++) {
        thread_pool.start(new RTWorker(*this));
    }
}

RayTracer::~RayTracer() {
    cancelling=true;
    thread_pool.waitForDone(-1);
    delete[] buffer;
    if (first_pass_buffer != nullptr) {
        delete[] first_pass_buffer;
    }
    if (debug_camera_used_) {
        debug_camera_used_->ClearDebugRays();
    }
}

int RayTracer::GetProgress() {
    if (thread_pool.waitForDone(1)) {
        return 100;
    }

    const unsigned int wc = (settings.width+THREAD_CHUNKSIZE-1)/THREAD_CHUNKSIZE;
    const unsigned int hc = (settings.height+THREAD_CHUNKSIZE-1)/THREAD_CHUNKSIZE;
    int complete = (100*next_render_index.fetchAndAddRelaxed(0))/(wc*hc);
    return std::min(complete,99);
}

double RayTracer::AspectRatio() {
    return ((double)settings.width)/((double)settings.height);
}


void RayTracer::ComputePixel(int i, int j, Camera* debug_camera) {
    // Calculate the normalized coordinates [0, 1]
    double x_corner = i * settings.pixel_size_x;
    double y_corner = j * settings.pixel_size_y;

    if (debug_camera) {
        if (debug_camera_used_) {
            debug_camera_used_->ClearDebugRays();
        }
        debug_camera_used_ = debug_camera;
        SampleCamera(x_corner, y_corner, settings.pixel_size_x, settings.pixel_size_y, debug_camera);
        return;
    }

    // Trace the ray!
    glm::vec3 color(0,0,0);

    switch (settings.random_mode)
    {
    case Camera::TRACERANDOM_DETERMINISTIC:
        switch (settings.samplecount_mode) {
        case Camera::TRACESAMPLING_CONSTANT:
            // Anti-aliasing
            // use setting.constant_samples_per_pixel to get the amount of samples of a pixel for anti-alasing
            {
                unsigned int num_samples = settings.constant_samples_per_pixel;
                int grid_size = (int) sqrt(num_samples);
                double subpixel_size_x = settings.pixel_size_x / grid_size;
                double subpixel_size_y = settings.pixel_size_y / grid_size;
                for (int i = 0; i < grid_size; i++) {
                    for (int j = 0; j < grid_size; j++) {
                        double cell_x_corner = x_corner + j * subpixel_size_x;
                        double cell_y_corner = y_corner + i * subpixel_size_y;
                        color += SampleCamera(cell_x_corner, cell_y_corner, subpixel_size_x, subpixel_size_y, debug_camera, false);
                    }
                }
                color /= num_samples;
            }
            break;
        default:
            break;
        }
        break;
    case Camera::TRACERANDOM_UNIFORM:
        switch (settings.samplecount_mode) {
        case Camera::TRACESAMPLING_CONSTANT:
            {
                unsigned int num_samples = settings.constant_samples_per_pixel;
                for (int i = 0; i < (int) settings.constant_samples_per_pixel; i++) {
                    color += SampleCamera(x_corner, y_corner, settings.pixel_size_x, settings.pixel_size_y, debug_camera, true);
                }
                color /= num_samples;
            }
            break;
        default:
            break;
        }
        break;
    case Camera::TRACERANDOM_STRATIFIED:
        switch (settings.samplecount_mode) {
        case Camera::TRACESAMPLING_CONSTANT:
            {
                unsigned int num_samples = settings.constant_samples_per_pixel;
                int grid_size = (int) sqrt(num_samples);
                double subpixel_size_x = settings.pixel_size_x / grid_size;
                double subpixel_size_y = settings.pixel_size_y / grid_size;
                for (int i = 0; i < grid_size; i++) {
                    for (int j = 0; j < grid_size; j++) {
                        double cell_x_corner = x_corner + j * subpixel_size_x;
                        double cell_y_corner = y_corner + i * subpixel_size_y;
                        color += SampleCamera(cell_x_corner, cell_y_corner, subpixel_size_x, subpixel_size_y, debug_camera, true);
                    }
                }
                color /= num_samples;
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }

    color = glm::clamp(color, 0.0f, 1.0f);

    // Set the pixel in the render buffer
    uint8_t* pixel = buffer + (i + j * settings.width) * 3;
    pixel[0] = (uint8_t)( 255.0f * color[0]);
    pixel[1] = (uint8_t)( 255.0f * color[1]);
    pixel[2] = (uint8_t)( 255.0f * color[2]);
}


glm::vec3 RayTracer::SampleCamera(double x_corner, double y_corner, double pixel_size_x, double pixel_size_y, Camera* debug_camera, bool random)
{
    double x = x_corner + pixel_size_x * (random ? ((double) rand() / RAND_MAX) : 0.5);
    double y = y_corner + pixel_size_y * (random ? ((double) rand() / RAND_MAX) : 0.5);

    glm::dvec3 point_on_focus_plane = settings.projection_origin + settings.projection_forward + (2.0*x-1.0)*settings.projection_right + (2.0*y-1.0)*settings.projection_up;

    double angle = 0.0;
    double radius = sqrt(0.0);

    if (settings.random_mode != Camera::TRACERANDOM_DETERMINISTIC) {
        // Monte Carlo: depth of field, sample eye position
        angle = 2 * M_PI * ((double) rand() / RAND_MAX);
        radius = sqrt((double) rand() / RAND_MAX);
    }

    glm::dvec3 origin = settings.projection_origin + radius * (sin(angle) * settings.aperture_up + cos(angle) * settings.aperture_right);

    glm::dvec3 dir = glm::normalize(point_on_focus_plane - origin);

    Ray camera_ray(origin, dir);

    return TraceRay(camera_ray, 0, RayType::camera, debug_camera);
}

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
// Depth is the number of times the ray has intersected an object.
glm::vec3 RayTracer::TraceRay(const Ray& r, int depth, RayType ray_type, Camera* debug_camera)
{
    Intersection i;

    if (debug_camera) {
        glm::dvec3 endpoint = r.at(1000);
        if (trace_scene.Intersect(r, i)) {
            endpoint = r.at(i.t);
            debug_camera->AddDebugRay(endpoint, endpoint+0.25*(glm::dvec3)i.normal, RayType::hit_normal);
        }
        debug_camera->AddDebugRay(r.position, endpoint, ray_type);
    }

    if (trace_scene.Intersect(r, i)) {
        // An intersection occured!
        Material* mat = i.GetMaterial();
        glm::vec3 kd = mat->Diffuse->GetColorUV(i.uv);
        glm::vec3 ks = mat->Specular->GetColorUV(i.uv);
        glm::vec3 ke = mat->Emissive->GetColorUV(i.uv);
        glm::vec3 kt = mat->Transmittence->GetColorUV(i.uv);
        float shininess = mat->Shininess;
        double index_of_refraction = mat->IndexOfRefraction;

        // Interpolated normal
        // Use this to when calculating geometry (entering object test, reflection, refraction, etc) or getting smooth shading (light direction test, etc)
        glm::dvec3 N = i.normal;
        glm::dvec3 Q = r.at(i.t);
        glm::dvec3 V = glm::normalize(-r.direction);
        // inside an object if normal pointing away from ray
        bool insideObject = glm::dot(-V, N) > 0.0;
        if (insideObject) {
            N = -N;
        }

        glm::vec3 color = ke;

        // Iterate over all light sources in the scene
        for (auto j = trace_scene.lights.begin(); j != trace_scene.lights.end(); j++) {
            TraceLight* trace_light = *j;
            Light* scene_light = trace_light->light;
            glm::dvec3 L;
            glm::dvec3 H;
            float a_dist = 1.0f;
            glm::vec3 a_shadow(1.0);

            // distance attenuation for attenuating lights
            if (auto attenuating_light = dynamic_cast<AttenuatingLight*>(scene_light)) {
                double distance = glm::length(glm::dvec3(trace_light->GetTransformPos()) - Q);
                a_dist = DistanceAttenuation(attenuating_light, distance);
            }

            if (auto point_light = dynamic_cast<PointLight*>(scene_light)) {
                // point light
                glm::dvec3 light_position = trace_light->GetTransformPos();
                L = glm::normalize(light_position - Q);

                if (settings.shadows && glm::dot(N, L) >= NORMAL_EPSILON) {
                    // only shoot shadow rays if not nullified by B = 0
                    Ray shadow_ray(Q, L);
                    a_shadow = ShadowAttenuation(shadow_ray, 0, light_position, debug_camera);
                }
            } else if (auto directional_light = dynamic_cast<DirectionalLight*>(scene_light)) {
                // directional light
                glm::vec3 light_direction = trace_light->GetTransformDirection();
                // light_direction is actually reversed
                L = glm::normalize(light_direction);

                if (settings.shadows && glm::dot(N, L) >= NORMAL_EPSILON) {
                    // only shoot shadow rays if not nullified by B = 0
                    Ray shadow_ray(Q, L);
                    a_shadow = ShadowAttenuation(shadow_ray, 0, shadow_ray.at(1000), debug_camera);
                }
            } else if (auto area_light = dynamic_cast<AreaLight*>(scene_light)) {
                // area light
                // follow Marschner-Shirley, randomly choose point in area: r = c + xi1 * a + xi2 * b
                glm::dvec3 point_light_position = trace_light->GetTransformPos();
                if (settings.random_mode != Camera::TRACERANDOM_DETERMINISTIC) {
                    // Monte Carlo: sample point light position from area
                    glm::vec3 c = ((trace_light->transform) * glm::vec4(-0.5, 0, -0.5, 1)).xyz;
                    glm::vec3 c1 = ((trace_light->transform) * glm::vec4(-0.5, 0, 0.5, 1)).xyz;
                    glm::vec3 c2 = ((trace_light->transform) * glm::vec4(0.5, 0, -0.5, 1)).xyz;
                    glm::vec3 a = c1 - c;
                    glm::vec3 b = c2 - c;
                    double xi1 = (double) rand() / RAND_MAX;
                    double xi2 = (double) rand() / RAND_MAX;
                    point_light_position = glm::dvec3(c) + xi1 * glm::dvec3(a) + xi2 * glm::dvec3(b);
                    // re-calculate distance attenuation
                    double distance = glm::length(point_light_position - Q);
                    a_dist = DistanceAttenuation(area_light, distance);
                }
                L = glm::normalize(point_light_position - Q);
                if (settings.shadows && glm::dot(N, L) > 0) {
                    Ray shadow_ray(Q, L);
                    a_shadow = ShadowAttenuation(shadow_ray, 0, point_light_position, debug_camera);
                }
            }

            H = glm::normalize(V + L);

            float B = glm::dot(N, L) < NORMAL_EPSILON ? 0.0f : 1.0f;
            float diffuseShade = glm::max(glm::dot(N, L), 0.0);
            float _shininess = shininess > 0 ? shininess : NORMAL_EPSILON;
            float specularShade = B * glm::pow(glm::max(glm::dot(H, N), 0.0), _shininess);

            glm::vec3 ambient = kd * scene_light->Ambient.GetRGB();
            // if (insideObject) {
            //     ambient *= kt;
            // }
            glm::vec3 diffuse = a_dist * a_shadow * diffuseShade * kd * scene_light->GetIntensity();
            glm::vec3 specular = a_dist * a_shadow * specularShade * ks * scene_light->GetIntensity();
            color += ambient + diffuse + specular;
        }

        // Test if the Reflections and Refractions checkboxes are enabled in the Render Cam UI.
        // Only calculate reflection/refraction if enabled.
        if (depth < (int) settings.max_depth) {
            if (settings.reflections && glm::length2(ks) >= RAY_EPSILON) {
                glm::dvec3 R = glm::normalize(2.0 * glm::dot(V, N) * N - V);
                if (settings.random_mode != Camera::TRACERANDOM_DETERMINISTIC) {
                    // Monte Carlo: glossy reflection
                    // calculate orthonormal basis
                    int smallest_comp_index = 0;
                    int smallest_comp_value = R.x;
                    if (R.y < smallest_comp_value) {
                        smallest_comp_index = 1;
                        smallest_comp_value = R.y;
                    }
                    if (R.z < smallest_comp_value) {
                        smallest_comp_index = 2;
                        smallest_comp_value = R.z;
                    }
                    glm::dvec3 T(R);
                    switch (smallest_comp_index)
                    {
                    case 0:
                        T.x = 1;
                        break;
                    case 1:
                        T.y = 1;
                        break;
                    case 2:
                        T.z = 1;
                    default:
                        break;
                    }
                    glm::dvec3 U = glm::normalize(glm::cross(R, T));
                    glm::dvec3 V = glm::normalize(glm::cross(R, U));
                    // TODO: use something else for a
                    double a = 1.0;
                    double xi1 = (double) rand() / RAND_MAX;
                    double xi2 = (double) rand() / RAND_MAX;
                    double u = - a / 2 + xi1 * a;
                    double v = - a / 2 + xi2 * a;
                    glm::dvec3 R_diffuse = glm::normalize(R + u * U + v * V);
                    Ray reflection_ray(Q, R_diffuse);
                    color += ks * TraceRay(reflection_ray, depth + 1, RayType::reflection, debug_camera);
                } else {
                    Ray reflection_ray(Q, R);
                    color += ks * TraceRay(reflection_ray, depth + 1, RayType::reflection, debug_camera);
                }
            }
            if (settings.diffuse_reflection && settings.random_mode != Camera::TRACERANDOM_DETERMINISTIC) {
                // Monte Carlo: diffuse reflection (?)
                double u = -1 + 2 * ((double) rand() / RAND_MAX); // u: [-1, 1]
                double theta = 2 * M_PI * ((double) rand() / RAND_MAX); // theta: [0, 2pi]
                glm::dvec3 e(glm::sqrt(1- u * u) * glm::cos(theta), glm::sqrt(1- u * u) * glm::sin(theta), u);
                glm::dvec3 R;
                if (glm::length(N + e) < NORMAL_EPSILON) {
                    R = N;
                } else {
                    R = glm::normalize(N + e);
                }
                double cos_theta = glm::dot(R, N);
                glm::vec3 brdf = glm::dvec3(ks) / M_PI;
                const double p = 1/(2*M_PI);
                Ray diffuse_reflection_ray(Q, R);
                color += brdf * TraceRay(diffuse_reflection_ray, depth + 1, RayType::diffuse_reflection, debug_camera) * (float) (cos_theta / p);
            }
            if (settings.refractions && glm::length2(kt) >= RAY_EPSILON) {
                double eta_i, eta_t;
                if (insideObject) {
                    eta_i = index_of_refraction;
                    eta_t = INDEX_OF_AIR;
                } else {
                    eta_i = INDEX_OF_AIR;
                    eta_t = index_of_refraction;
                }
                double eta = eta_i / eta_t;
                double cos_theta_i = glm::dot(N, V);
                double cos_theta_t_sq = 1 - eta * eta * (1 - cos_theta_i * cos_theta_i);
                if (cos_theta_t_sq >= NORMAL_EPSILON) {
                    // not total internal reflection
                    double cos_theta_t = glm::sqrt(cos_theta_t_sq);
                    glm::dvec3 T = glm::normalize((eta * cos_theta_i - cos_theta_t) * N - eta * V);
                    // refracted way entering surface, continue
                    Ray refraction_ray(Q, T);
                    color += kt * TraceRay(refraction_ray, depth + 1, RayType::refraction, debug_camera);
                }
            }
        }
        return color;
    } else {
        // No intersection. This ray travels to infinity, so we color it according to the background color,
        // which in this (simple) case is just black.
        glm::vec3 background_color = glm::vec3(0, 0, 0);
        if (!settings.skybox) {
            return background_color;
        }
        // EXTRA CREDIT: Use environment mapping to determine the color instead
        // faces: "ft","bk","up","dn","rt","lf"
        int max_component_index = 0;
        double max_component_abs = glm::abs(r.direction.x);
        if (glm::abs(r.direction.y) > max_component_abs) {
            max_component_index = 1;
            max_component_abs = glm::abs(r.direction.y);
        }
        if (glm::abs(r.direction.z) > max_component_abs) {
            max_component_index = 2;
            max_component_abs = glm::abs(r.direction.z);
        }
        unsigned int resolution = settings.skybox->GetResolution();
        int face;
        float face_x, face_y;
        switch (max_component_index)
        {
        case 0:
            if (r.direction.x >= 0) {
                // right
                face = 4;
                face_x = ((r.direction.z / max_component_abs) + 1.0) / 2.0 * resolution;
                face_y = (-(r.direction.y / max_component_abs) + 1.0) / 2.0 * resolution;
            } else {
                // left
                face = 5;
                face_x = (-(r.direction.z / max_component_abs) + 1.0) / 2.0 * resolution;
                face_y = (-(r.direction.y / max_component_abs) + 1.0) / 2.0 * resolution;
            }
            break;
        case 1:
            if (r.direction.y >= 0) {
                // top
                face = 2;
                face_x = ((r.direction.x / max_component_abs) + 1.0) / 2.0 * resolution;
                face_y = (-(r.direction.z / max_component_abs) + 1.0) / 2.0 * resolution;
            } else {
                // bottom
                face = 3;
                face_x = ((r.direction.x / max_component_abs) + 1.0) / 2.0 * resolution;
                face_y = ((r.direction.z / max_component_abs) + 1.0) / 2.0 * resolution;
            }
            break;
        case 2:
            if (r.direction.z >= 0) {
                // front
                face = 0;
                face_x = (-(r.direction.x / max_component_abs) + 1.0) / 2.0 * resolution;
                face_y = (-(r.direction.y / max_component_abs) + 1.0) / 2.0 * resolution;
            } else {
                // back
                face = 1;
                face_x = ((r.direction.x / max_component_abs) + 1.0) / 2.0 * resolution;
                face_y = (-(r.direction.y / max_component_abs) + 1.0) / 2.0 * resolution;
            }
        default:
            break;
        }
        // bilinear interpolation
        const unsigned char *image = settings.skybox->GetFace(face);
        int x1 = (int) floor(face_x);
        int x2 = (int) ceil(face_x);
        int y1 = (int) floor(face_y);
        int y2 = (int) ceil(face_y);
        x2 = x2 >= resolution ? resolution - 1 : x2;
        y2 = y2 >= resolution ? resolution - 1 : y2;
        const unsigned int CHANNELS = 4; // RGBA
        int tl_index = y2 * resolution * CHANNELS + x1 * CHANNELS;
        int tr_index = y2 * resolution * CHANNELS + x2 * CHANNELS;
        int bl_index = y1 * resolution * CHANNELS + x1 * CHANNELS;
        int br_index = y1 * resolution * CHANNELS + x2 * CHANNELS;
        glm::vec4 tl(image[tl_index], image[tl_index+1], image[tl_index+2], image[tl_index+3]);
        glm::vec4 tr(image[tr_index], image[tr_index+1], image[tr_index+2], image[tr_index+3]);
        glm::vec4 bl(image[bl_index], image[bl_index+1], image[bl_index+2], image[bl_index+3]);
        glm::vec4 br(image[br_index], image[br_index+1], image[br_index+2], image[br_index+3]);
        glm::vec4 tm = (face_x - x1) * tr + (x2 - face_x) * tl;
        glm::vec4 bm = (face_x - x1) * br + (x2 - face_x) * bl;
        glm::vec4 color = (face_y - y1) * tm + (y2 - face_y) * bm;
        glm::vec4 normalized_color = glm::clamp(color / 255.0f, 0.0f, 1.0f);
        return normalized_color.xyz;
    }
}

double RayTracer::DistanceAttenuation(AttenuatingLight* attenuating_light, double distance)
{
    double a = attenuating_light->AttenA.Get();
    double b = attenuating_light->AttenB.Get();
    double c = attenuating_light->AttenC.Get();
    double a_dist_r = a * distance * distance + b * distance + c;
    double a_dist = (a_dist_r != 0) ? (1.0 / a_dist_r) : 1.0;
    return glm::min(1.0, a_dist);
}

glm::vec3 RayTracer::ShadowAttenuation(const Ray& shadow_ray, int depth, glm::vec3 light_position, Camera* debug_camera)
{
    Intersection i;
    if (debug_camera) {
        glm::dvec3 endpoint = light_position;
        if (trace_scene.Intersect(shadow_ray, i)) {
            endpoint = shadow_ray.at(i.t);
        }
        debug_camera->AddDebugRay(shadow_ray.position, endpoint, RayType::shadow);
    }

    if (trace_scene.Intersect(shadow_ray, i)) {
        if (glm::length(shadow_ray.direction * i.t) > glm::distance(glm::dvec3(light_position), shadow_ray.position)) {
            // already past the light, this intersection doesn't block light
            return glm::vec3(1.0);
        }
        if (!settings.translucent_shadows) {
            return glm::vec3(0.0);
        }
        Material* mat = i.GetMaterial();
        glm::vec3 kt = mat->Transmittence->GetColorUV(i.uv);
        if (glm::length2(kt) < RAY_EPSILON) {
            return kt;
        }
        glm::dvec3 Q = shadow_ray.at(i.t);
        Ray next_shadow_ray(Q, shadow_ray.direction);
        kt *= ShadowAttenuation(next_shadow_ray, depth + 1, light_position, debug_camera);
        return kt;
    } else {
        // nothing is in the way
        return glm::vec3(1.0);
    }
}

// Multi-Threading
RTWorker::RTWorker(RayTracer &tracer_) :
    tracer(tracer_) { }

void RTWorker::run() {
    // Dimensions, in chunks
    const unsigned int wc = (tracer.settings.width+THREAD_CHUNKSIZE-1)/THREAD_CHUNKSIZE;
    const unsigned int hc = (tracer.settings.height+THREAD_CHUNKSIZE-1)/THREAD_CHUNKSIZE;

    unsigned int x, y;
    while (!tracer.cancelling) {
        unsigned int idx = tracer.next_render_index.fetchAndAddRelaxed(1);
        unsigned int x = (idx%wc)*THREAD_CHUNKSIZE;
        unsigned int y = (idx/wc)*THREAD_CHUNKSIZE;
        if (y >= tracer.settings.height) break;
        unsigned int maxX = std::min(x + THREAD_CHUNKSIZE, tracer.settings.width);
        unsigned int maxY = std::min(y + THREAD_CHUNKSIZE, tracer.settings.height);

        for(unsigned int yy = y; yy < maxY && !tracer.cancelling; yy++) {
            for(unsigned int xx = x; xx < maxX && !tracer.cancelling; xx++) {
                tracer.ComputePixel(xx, yy);
            }
        }
    }
}
