#include "triangleface.h"

TriangleFace::TriangleFace(glm::vec3 a_, glm::vec3 b_, glm::vec3 c_, glm::vec3 a_n_, glm::vec3 b_n_, glm::vec3 c_n_, glm::vec2 a_uv_, glm::vec2 b_uv_, glm::vec2 c_uv_, bool use_per_vertex_normals_) :
    a(a_), b(b_), c(c_), a_n(a_n_), b_n(b_n_), c_n(c_n_), a_uv(a_uv_), b_uv(b_uv_), c_uv(c_uv_), use_per_vertex_normals(use_per_vertex_normals_)
{
    local_bbox.reset(new BoundingBox(glm::min(a,glm::min(b,c)),glm::max(a,glm::max(b,c))));
}

bool TriangleFace::IntersectLocal(const Ray &r, Intersection &i)
{
    // Triangle intersection code.

    // Note that you are only intersecting a single triangle, and the vertices
    // of the triangle are supplied to you by the trimesh class.
    //
    // use_per_vertex_normals tells you if the triangle has per-vertex normals.
    // If it does, you should compute and use the Phong-interpolated normal at the intersection point.
    // If it does not, you should use the normal of the triangle's supporting plane.
    glm::dvec3 e = r.position;
    glm::dvec3 d = r.direction;

    glm::dvec3 n = glm::normalize(glm::cross(b - a, c - a));
    double k = glm::dot(n, a);

    // plane equation: n.x = k

    if (glm::abs(glm::dot(n, d)) < NORMAL_EPSILON) {
        // ray parallel to plane, no intersection
        return false;
    }

    double t = (k - glm::dot(n, e)) / glm::dot(n, d);
    if (t < RAY_EPSILON) {
        return false;
    }

    glm::dvec3 q = r.at(t);
   
    double test_ab = glm::dot(glm::cross(b - a, q - a), n);
    double test_bc = glm::dot(glm::cross(c - b, q - b), n);
    double test_ca = glm::dot(glm::cross(a - c, q - c), n);

    if (test_ab < -EDGE_EPSILON || test_bc < -EDGE_EPSILON || test_ca < -EDGE_EPSILON) {
        // Q outside triangle
        return false;
    }

    double denom = glm::dot(glm::cross(b - a, c - a), n);
    double alpha = test_bc / denom;
    double beta = test_ca / denom;
    double gamma = test_ab / denom;

    // If the ray r intersects the triangle abc:
    // 1. put the hit parameter in i.t
    i.t = t;
    // 2. put the normal in i.normal
    if (use_per_vertex_normals) {
        glm::dvec3 na = glm::dvec3(a_n);
        glm::dvec3 nb = glm::dvec3(b_n);
        glm::dvec3 nc = glm::dvec3(c_n);
        i.normal = glm::normalize(alpha * na + beta * nb + gamma * nc);
    } else {
        i.normal = n;
    }
    // 3. put the texture coordinates in i.uv
    glm::dvec2 uva = glm::dvec2(a_uv);
    glm::dvec2 uvb = glm::dvec2(b_uv);
    glm::dvec2 uvc = glm::dvec2(c_uv);
    i.uv = alpha * uva + beta * uvb + gamma * uvc;
    // and return true
    return true;
}
