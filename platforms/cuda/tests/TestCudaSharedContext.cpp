/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaTests.h"
#include "openmm/Context.h"
#include "openmm/CustomCVForce.h"
#include "openmm/GBSAOBCForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

static System* createSystem(vector<Vec3>& positions, bool withCollectiveVariable = false) {
    System* system = new System();
    NonbondedForce* nonbonded = new NonbondedForce();
    GBSAOBCForce* gb = new GBSAOBCForce();
    HarmonicBondForce* bonds = new HarmonicBondForce();
    for (int i = 0; i < 500; i++) {
        double charge = (i%2 == 0 ? 0.2 : -0.2);
        system->addParticle(12.0);
        nonbonded->addParticle(charge, 0.32, 0.4);
        gb->addParticle(charge, 0.15, 0.8);
        positions.push_back(Vec3((i%10)*0.41, ((i/10)%10)*0.43, (i/100)*0.39));
        if (i%10 != 0)
            bonds->addBond(i-1, i, 0.4, 1000.0);
    }
    system->addForce(nonbonded);
    system->addForce(gb);
    if (withCollectiveVariable) {
        // CustomCVForce evaluates its variables in a linked inner Context.
        CustomCVForce* cv = new CustomCVForce("2*bonds");
        cv->addCollectiveVariable("bonds", bonds);
        system->addForce(cv);
    }
    else
        system->addForce(bonds);
    return system;
}

static State evaluate(const System& system, const vector<Vec3>& positions) {
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    return context.getState(State::Energy | State::Forces);
}

static size_t freeDeviceMemory() {
    string index = platform.getPropertyDefaultValue("DeviceIndex");
    CUdevice device;
    CUcontext context;
    size_t free, total;
    cuDeviceGet(&device, index.empty() ? 0 : atoi(index.c_str()));
    cuDevicePrimaryCtxRetain(&context, device);
    cuCtxPushCurrent(context);
    cuMemGetInfo(&free, &total);
    cuCtxPopCurrent(&context);
    cuDevicePrimaryCtxRelease(device);
    return free;
}

void testSharedContextGivesIdenticalResults() {
    vector<Vec3> positions;
    System* system = createSystem(positions);
    unsetenv("OPENMM_CUDA_SHARED_CONTEXT");
    State expected = evaluate(*system, positions);
    setenv("OPENMM_CUDA_SHARED_CONTEXT", "1", 1);
    State actual = evaluate(*system, positions);
    unsetenv("OPENMM_CUDA_SHARED_CONTEXT");
    ASSERT_EQUAL(expected.getPotentialEnergy(), actual.getPotentialEnergy());
    for (int i = 0; i < system->getNumParticles(); i++) {
        ASSERT_EQUAL_VEC(expected.getForces()[i], actual.getForces()[i], 0.0);
    }
    delete system;
}

void testSharedContextDoesNotLeak(bool withCollectiveVariable) {
    vector<Vec3> positions;
    System* system = createSystem(positions, withCollectiveVariable);
    setenv("OPENMM_CUDA_SHARED_CONTEXT", "1", 1);
    evaluate(*system, positions);
    size_t before = freeDeviceMemory();
    for (int i = 0; i < 200; i++)
        evaluate(*system, positions);
    size_t after = freeDeviceMemory();
    unsetenv("OPENMM_CUDA_SHARED_CONTEXT");
    ASSERT(before < after + 16*1024*1024);
    delete system;
}

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testSharedContextGivesIdenticalResults();
        testSharedContextDoesNotLeak(false);
        testSharedContextDoesNotLeak(true);
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
