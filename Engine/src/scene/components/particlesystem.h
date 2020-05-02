/****************************************************************************
 * Copyright Â©2017 Brian Curless.  All rights reserved.  Permission is hereby
 * granted to students registered for University of Washington CSE 457 or CSE
 * 557 for use solely during Autumn Quarter 2017 for purposes of the course.
 * No other use, copying, distribution, or modification is permitted without
 * prior written consent. Copyrights for third-party components of this work
 * must be honored.  Instructors interested in reusing these course materials
 * should contact the author.
 ****************************************************************************/
#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <scene/components/component.h>
#include <resource/material.h>

struct Particle {
public:
    float Mass;
    glm::vec3 Position;
    glm::vec3 Velocity;
    glm::vec3 Rotation;

    Particle(float mass_ = 1.0f, glm::vec3 position_ = glm::vec3(0,0,0), glm::vec3 velocity_ = glm::vec3(0,0,0), glm::vec3 rotation_ = glm::vec3(0,0,0)) :
        Mass(mass_), Position(position_), Velocity(velocity_), Rotation(rotation_) { }
};

class Force {
public:
    virtual glm::vec3 GetForce(Particle& p) const = 0;
};

class ConstantForce : public Force {
public:
    ConstantForce(glm::vec3 force) : force_(force) { }
    virtual glm::vec3 GetForce(Particle &p) const override {
        return p.Mass * force_;
    }
private:
    glm::vec3 force_;
};

class ParticleSystem : public Component {
public:
    ResourceProperty<Material> ParticleMaterial;
    DoubleProperty Period;
    DoubleProperty Restitution;

    ParticleSystem();

    void UpdateModelMatrix(glm::mat4 model_matrix);
    void EmitParticles();
    std::vector<Particle*> GetParticles();
    void StartSimulation();
    void UpdateSimulation(float delta_t, const std::vector<std::pair<SceneObject*, glm::mat4>>& colliders);
    void StopSimulation();
    void ResetSimulation();
    bool IsSimulating();

protected:
    static const unsigned int MAX_PARTICLES = 100;
    unsigned int num_particles_;
    unsigned int particle_index_;
    glm::mat4 model_matrix_;
    double time_to_emit_;
    bool simulating_;
    std::vector<std::unique_ptr<Particle>> particles_;
    std::vector<std::unique_ptr<Force>> forces_;
};


#endif // PARTICLESYSTEM_H