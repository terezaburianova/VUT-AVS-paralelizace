/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Tereza Burianova <xburia28@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    15 Dec 2022
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalCnt = 0;
    #pragma omp parallel
    #pragma omp single
    totalCnt = processChild(Vec3_t<float>(0,0,0), field, mGridSize);
    return totalCnt;
}

unsigned TreeMeshBuilder::processChild(const Vec3_t<float> &offset, const ParametricScalarField &field, size_t edgeSize)
{
    unsigned totalCnt = 0;

    Vec3_t<float> positions{
        (offset.x + edgeSize/2.F)*mGridResolution,
        (offset.y + edgeSize/2.F)*mGridResolution,
        (offset.z + edgeSize/2.F)*mGridResolution
    };

    const float cond = mIsoLevel + (sqrt(3.F) * mGridResolution * edgeSize)/2.0;
    if (evaluateFieldAt(positions, field) > cond) return 0;
    if (edgeSize <= cutoff) {
        return buildCube(offset, field);
    }
    
    for (Vec3_t<float> combination : sc_vertexNormPos) {
        #pragma omp task shared(totalCnt)
        {
            const Vec3_t<float> newOffset(
                offset.x + combination.x * edgeSize/2.F,
                offset.y + combination.y * edgeSize/2.F,
                offset.z + combination.z * edgeSize/2.F
            );
            unsigned cnt = processChild(newOffset, field, edgeSize/2.F);
            #pragma omp atomic
            totalCnt += cnt;
        }
    }
    #pragma omp taskwait
    return totalCnt;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}
