/****************************************************************************
 * Copyright ©2017 Brian Curless.  All rights reserved.  Permission is hereby
 * granted to students registered for University of Washington CSE 457 or CSE
 * 557 for use solely during Autumn Quarter 2017 for purposes of the course.
 * No other use, copying, distribution, or modification is permitted without
 * prior written consent. Copyrights for third-party components of this work
 * must be honored.  Instructors interested in reusing these course materials
 * should contact the author.
 ****************************************************************************/
#include "particlesystem.h"
#include <scene/sceneobject.h>

REGISTER_COMPONENT(ParticleSystem, ParticleSystem)

ParticleSystem::ParticleSystem() :
    ParticleMaterial(AssetType::Material),
    Period(0.1f, 0.0f, 1.0f, 0.01f),
    Restitution(1.0f, 0.0f, 2.0f, 0.1f),
    simulating_(false)
{
    AddProperty("Material", &ParticleMaterial);
    AddProperty("Period (s)", &Period);
    AddProperty("Restitution", &Restitution);

}

void ParticleSystem::UpdateModelMatrix(glm::mat4 model_matrix) {
   model_matrix_ = model_matrix;
}

void ParticleSystem::EmitParticles() {
    if (!simulating_) return;


    // Create some particles!
    // Particles should be created in world space.
    // For performance reasons, limit the amount of particles that exist at the same time
    // to some finite amount (MAX_PARTICLES). Either delete or recycle old particles as needed.

    // Reset the time
    time_to_emit_ = Period.Get();
}

std::vector<Particle*> ParticleSystem::GetParticles() {
    // Return a vector of particles (used by renderer to draw them)
    return std::vector<Particle*>(); // Return empty vector for now
}

void ParticleSystem::StartSimulation() {
    simulating_ = true;
    ResetSimulation();
}

void ParticleSystem::UpdateSimulation(float delta_t, const std::vector<std::pair<SceneObject*, glm::mat4>>& colliders) {
    if (!simulating_) return;

    // Emit Particles
    time_to_emit_ -= delta_t;
    if (time_to_emit_ <= 0.0) EmitParticles();

    // For each particle ...
    //      Calculate forces
    //      Solve the system of forces using Euler's method
    //      Update the particle
    //      Check for and handle collisions

    // Collision code might look something like this:
    for (auto& kv : colliders) {
        SceneObject* collider_object = kv.first;
        glm::mat4 collider_model_matrix = kv.second;
        if (SphereCollider* sphere_collider = collider_object->GetComponent<SphereCollider>()) {
            // Check for Sphere Collision
        } else if (PlaneCollider* plane_collider = collider_object->GetComponent<PlaneCollider>()) {
            // Check for Plane Collision
        }
    }
}

void ParticleSystem::StopSimulation() {
    simulating_ = false;
}

void ParticleSystem::ResetSimulation() {
    // Clear all particles
    time_to_emit_ = Period.Get();
}

bool ParticleSystem::IsSimulating() {
    return simulating_;
}
