/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Evan Pretti                                                       *
 * Contributors:                                                              *
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

#include "openmm/common/CommonMinimizeKernel.h"
#include "openmm/common/ContextSelector.h"
#include "CommonKernelSources.h"
#include <map>

using namespace OpenMM;
using namespace std;

// Constants for constraint handling.

const double CommonMinimizeKernel::minConstraintTol = 1e-4;
const double CommonMinimizeKernel::kRestraintScale = 100;
const double CommonMinimizeKernel::prevMaxErrorInit = 1e10;
const double CommonMinimizeKernel::kRestraintScaleUp = 10;
const double CommonMinimizeKernel::constraintTolScale = 100;

// Constants for L-BFGS.

const double CommonMinimizeKernel::fTol = 1e-4;
const double CommonMinimizeKernel::wolfeParam = 0.9;
const double CommonMinimizeKernel::stepScaleDown = 0.5;
const double CommonMinimizeKernel::stepScaleUp = 2.1;
const double CommonMinimizeKernel::minStep = 1e-20;
const double CommonMinimizeKernel::maxStep = 1e20;
const int CommonMinimizeKernel::numVectors = 6;
const int CommonMinimizeKernel::maxLineSearchIterations = 40;

CommonMinimizeKernel::~CommonMinimizeKernel() {
    if (cpuContext != NULL) {
        delete cpuContext;
    }
}

void CommonMinimizeKernel::initialize(const System& system) {
    numParticles = system.getNumParticles();
    numVariables = numParticles * 3;
    numConstraints = system.getNumConstraints();

    hostPositions.resize(numParticles);
    hostX.resize(numVariables);
    hostGrad.resize(numVariables);
    hostConstraintIndices.resize(numConstraints);
    hostConstraintDistances.resize(numConstraints);

    for (int i = 0; i < numConstraints; i++) {
        system.getConstraintParameters(i, hostConstraintIndices[i].x, hostConstraintIndices[i].y, hostConstraintDistances[i]);
    }
}

void CommonMinimizeKernel::execute(ContextImpl& context, double tolerance, int maxIterations, MinimizationReporter* reporter) {
    ContextSelector selector(cc);

    if (!isSetup) {
        // Load system and integrator information.

        mixedIsDouble = cc.getUseMixedPrecision() || cc.getUseDoublePrecision();
        if (mixedIsDouble && !cc.getSupports64BitGlobalAtomics()) {
            throw OpenMMException("Double precision is not supported on devices that do not support 64 bit atomic operations");
        }
        elementSize = mixedIsDouble ? sizeof(double) : sizeof(float);
        threadBlockSize = cc.getMaxThreadBlockSize();
        pinnedMemory = cc.getPinnedBuffer();

        // Initialize all device-side arrays and compile kernels.

        setup(context);
        isSetup = true;
    }

    const Integrator& integrator = context.getIntegrator();
    forceGroups = integrator.getIntegrationForceGroups();
    constraintTol = integrator.getConstraintTolerance();

    this->tolerance = tolerance * sqrt((double) numParticles);
    this->maxIterations = maxIterations;
    this->reporter = reporter;

    if (mixedIsDouble) {
        getDiffKernel->setArg(11, this->tolerance);
    }
    else {
        getDiffKernel->setArg(11, (float) this->tolerance);
    }

    double workingConstraintTol = max(minConstraintTol, constraintTol);
    kRestraint = kRestraintScale / workingConstraintTol;
    context.applyConstraints(workingConstraintTol);

    recordInitialPosKernel->execute(numParticles);

    double prevMaxError1 = prevMaxErrorInit, prevMaxError2 = prevMaxErrorInit;
    while (true) {
        lbfgs(context);

        if (!numConstraints) {
            // There are no constraints, so we are finished.
            break;
        }

        getConstraintErrorKernel->execute(threadBlockSize, threadBlockSize);
        double maxError = downloadReturnValueSync();
        if (maxError <= workingConstraintTol) {
            // All constraints are satisfied.
            break;
        }
        restorePosKernel->setArg(2, xInit);
        restorePosKernel->execute(numParticles);
        if (maxError >= prevMaxError2) {
            // Further tightening the springs doesn't seem to be helping, so just give up.
            break;
        }
        prevMaxError2 = prevMaxError1;
        prevMaxError1 = maxError;
        kRestraint *= kRestraintScaleUp;
        if (maxError > constraintTolScale * workingConstraintTol) {
            // We've gotten far enough from a valid state that we might have
            // trouble getting back, so reset to the original positions.
            xInit.copyTo(x);
        }
    }

    if (constraintTol < workingConstraintTol) {
        context.applyConstraints(constraintTol);
    }
}

void CommonMinimizeKernel::setup(ContextImpl& context) {
    // Initialize arrays.

    if (numConstraints) {
        constraintIndices.initialize<mm_int2>(cc, numConstraints, "constraintIndices");
        constraintDistances.initialize(cc, numConstraints, elementSize, "constraintDistances");
    }
    xInit.initialize(cc, numVariables, elementSize, "xInit");
    x.initialize(cc, numVariables, elementSize, "x");
    xPrev.initialize(cc, numVariables, elementSize, "xPrev");
    grad.initialize(cc, numVariables, elementSize, "grad");
    gradPrev.initialize(cc, numVariables, elementSize, "gradPrev");
    dir.initialize(cc, numVariables, elementSize, "dir");
    alpha.initialize(cc, numVectors + 1, elementSize, "alpha");
    scale.initialize(cc, numVectors, elementSize, "scale");
    xDiff.initialize(cc, numVectors * numVariables, elementSize, "xDiff");
    gradDiff.initialize(cc, numVectors * numVariables, elementSize, "gradDiff");
    returnFlag.initialize<int>(cc, 1, "returnFlag");
    returnValue.initialize(cc, 1, elementSize, "returnValue");
    gradNorm.initialize(cc, 1, elementSize, "gradNorm");
    lineSearchData.initialize(cc, 3, elementSize, "lineSearchData");
    lineSearchDataBackup.initialize(cc, 3, elementSize, "lineSearchDataBackup");

    // Allocate scratch buffers for deterministic two-pass reductions.  Size is
    // the maximum number of thread blocks any producer kernel can launch with,
    // matching cc.getNumThreadBlocks() (the same cap CudaContext, HipContext,
    // and OpenCLContext apply inside executeKernel()).
    maxReductionBlocks = cc.getNumThreadBlocks();
    reductionPartials1.initialize(cc, maxReductionBlocks, elementSize, "reductionPartials1");
    reductionPartials2.initialize(cc, maxReductionBlocks, elementSize, "reductionPartials2");

    // Compile kernels and set arguments.

    map<string, string> defines;
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(threadBlockSize);
    defines["NUM_VECTORS"] = cc.intToString(numVectors);
    defines["LBFGS_FTOL"] = cc.doubleToString(fTol);
    defines["LBFGS_WOLFE"] = cc.doubleToString(wolfeParam);
    defines["LBFGS_SCALE_DOWN"] = cc.doubleToString(stepScaleDown);
    defines["LBFGS_SCALE_UP"] = cc.doubleToString(stepScaleUp);
    defines["LBFGS_MIN_STEP"] = cc.doubleToString(minStep);
    defines["LBFGS_MAX_STEP"] = cc.doubleToString(maxStep);

    ComputeProgram program = cc.compileProgram(CommonKernelSources::minimize, defines);

    recordInitialPosKernel = program->createKernel("recordInitialPos");
    recordInitialPosKernel->addArg(cc.getPosq());
    recordInitialPosKernel->addArg(cc.getAtomIndexArray());
    recordInitialPosKernel->addArg(xInit);
    recordInitialPosKernel->addArg(x);
    recordInitialPosKernel->addArg(numParticles);
    if (cc.getUseMixedPrecision()) {
        recordInitialPosKernel->addArg(cc.getPosqCorrection());
    }

    restorePosKernel = program->createKernel("restorePos");
    restorePosKernel->addArg(cc.getPosq());
    restorePosKernel->addArg(cc.getAtomIndexArray());
    restorePosKernel->addArg(); // x (could also be launched with xInit)
    restorePosKernel->addArg(returnValue);
    restorePosKernel->addArg(numParticles);
    if (cc.getUseMixedPrecision()) {
        restorePosKernel->addArg(cc.getPosqCorrection());
    }

    convertForcesKernel = program->createKernel("convertForces");
    convertForcesKernel->addArg(cc.getVelm());
    convertForcesKernel->addArg(cc.getLongForceBuffer());
    convertForcesKernel->addArg(cc.getAtomIndexArray());
    convertForcesKernel->addArg(grad);
    convertForcesKernel->addArg(returnValue);
    convertForcesKernel->addArg(numParticles);
    convertForcesKernel->addArg(cc.getPaddedNumAtoms());

    if (numConstraints) {
        getConstraintEnergyForcesKernel = program->createKernel("getConstraintEnergyForces");
        getConstraintEnergyForcesKernel->addArg(cc.getLongForceBuffer());
        getConstraintEnergyForcesKernel->addArg(cc.getAtomIndexArray());
        getConstraintEnergyForcesKernel->addArg(constraintIndices);
        getConstraintEnergyForcesKernel->addArg(constraintDistances);
        getConstraintEnergyForcesKernel->addArg(x);
        getConstraintEnergyForcesKernel->addArg(reductionPartials1);
        getConstraintEnergyForcesKernel->addArg(cc.getPaddedNumAtoms());
        getConstraintEnergyForcesKernel->addArg(numConstraints);
        getConstraintEnergyForcesKernel->addArg(); // kRestraint

        getConstraintErrorKernel = program->createKernel("getConstraintError");
        getConstraintErrorKernel->addArg(constraintIndices);
        getConstraintErrorKernel->addArg(constraintDistances);
        getConstraintErrorKernel->addArg(x);
        getConstraintErrorKernel->addArg(returnValue);
        getConstraintErrorKernel->addArg(numConstraints);
    }

    initializeDirKernel = program->createKernel("initializeDir");
    initializeDirKernel->addArg(grad);
    initializeDirKernel->addArg(dir);
    initializeDirKernel->addArg(gradNorm);
    initializeDirKernel->addArg(lineSearchData);
    initializeDirKernel->addArg(numVariables);

    gradNormKernel = program->createKernel("gradNorm");
    gradNormKernel->addArg(grad);
    gradNormKernel->addArg(gradNorm);
    gradNormKernel->addArg(numVariables);

    getDiffKernel = program->createKernel("getDiff");
    getDiffKernel->addArg(x);
    getDiffKernel->addArg(xPrev);
    getDiffKernel->addArg(grad);
    getDiffKernel->addArg(gradPrev);
    getDiffKernel->addArg(scale);
    getDiffKernel->addArg(xDiff);
    getDiffKernel->addArg(gradDiff);
    getDiffKernel->addArg(returnFlag);
    getDiffKernel->addArg(returnValue);
    getDiffKernel->addArg(gradNorm);
    getDiffKernel->addArg(numVariables);
    getDiffKernel->addArg(); // tolerance
    getDiffKernel->addArg(); // end
    getDiffKernel->addArg(); // largeGrad

    getScaleKernel = program->createKernel("getScale");
    getScaleKernel->addArg(alpha);
    getScaleKernel->addArg(scale);
    getScaleKernel->addArg(xDiff);
    getScaleKernel->addArg(gradDiff);
    getScaleKernel->addArg(returnFlag);
    getScaleKernel->addArg(returnValue);
    getScaleKernel->addArg(reductionPartials1);
    getScaleKernel->addArg(reductionPartials2);
    getScaleKernel->addArg(numVariables);
    getScaleKernel->addArg(); // end
    getScaleKernel->addArg(); // largeGrad

    reinitializeDirKernel = program->createKernel("reinitializeDir");
    reinitializeDirKernel->addArg(grad);
    reinitializeDirKernel->addArg(dir);
    reinitializeDirKernel->addArg(alpha);
    reinitializeDirKernel->addArg(scale);
    reinitializeDirKernel->addArg(xDiff);
    reinitializeDirKernel->addArg(returnFlag);
    reinitializeDirKernel->addArg(returnValue);
    reinitializeDirKernel->addArg(reductionPartials1);
    reinitializeDirKernel->addArg(numVariables);
    reinitializeDirKernel->addArg(); // vectorIndex
    reinitializeDirKernel->addArg(); // largeGrad

    updateDirAlphaKernel = program->createKernel("updateDirAlpha");
    updateDirAlphaKernel->addArg(dir);
    updateDirAlphaKernel->addArg(alpha);
    updateDirAlphaKernel->addArg(scale);
    updateDirAlphaKernel->addArg(xDiff);
    updateDirAlphaKernel->addArg(gradDiff);
    updateDirAlphaKernel->addArg(returnFlag);
    updateDirAlphaKernel->addArg(reductionPartials1);
    updateDirAlphaKernel->addArg(numVariables);
    updateDirAlphaKernel->addArg(); // vectorIndex

    scaleDirKernel = program->createKernel("scaleDir");
    scaleDirKernel->addArg(dir);
    scaleDirKernel->addArg(alpha);
    scaleDirKernel->addArg(scale);
    scaleDirKernel->addArg(gradDiff);
    scaleDirKernel->addArg(returnFlag);
    scaleDirKernel->addArg(returnValue);
    scaleDirKernel->addArg(reductionPartials1);
    scaleDirKernel->addArg(numVariables);
    scaleDirKernel->addArg(); // vectorIndex

    updateDirBetaKernel = program->createKernel("updateDirBeta");
    updateDirBetaKernel->addArg(dir);
    updateDirBetaKernel->addArg(alpha);
    updateDirBetaKernel->addArg(scale);
    updateDirBetaKernel->addArg(xDiff);
    updateDirBetaKernel->addArg(gradDiff);
    updateDirBetaKernel->addArg(returnFlag);
    updateDirBetaKernel->addArg(reductionPartials1);
    updateDirBetaKernel->addArg(numVariables);
    updateDirBetaKernel->addArg(); // vectorIndex
    updateDirBetaKernel->addArg(); // vectorIndexAlpha

    updateDirFinalKernel = program->createKernel("updateDirFinal");
    updateDirFinalKernel->addArg(dir);
    updateDirFinalKernel->addArg(alpha);
    updateDirFinalKernel->addArg(xDiff);
    updateDirFinalKernel->addArg(returnFlag);
    updateDirFinalKernel->addArg(lineSearchData);
    updateDirFinalKernel->addArg(numVariables);
    updateDirFinalKernel->addArg(); // vectorIndex
    updateDirFinalKernel->addArg(); // vectorIndexAlpha

    lineSearchSetupKernel = program->createKernel("lineSearchSetup");
    lineSearchSetupKernel->addArg(x);
    lineSearchSetupKernel->addArg(xPrev);
    lineSearchSetupKernel->addArg(grad);
    lineSearchSetupKernel->addArg(gradPrev);
    lineSearchSetupKernel->addArg(dir);
    lineSearchSetupKernel->addArg(returnFlag);
    lineSearchSetupKernel->addArg(gradNorm);
    lineSearchSetupKernel->addArg(lineSearchData);
    lineSearchSetupKernel->addArg(reductionPartials1);
    lineSearchSetupKernel->addArg(numVariables);

    lineSearchStepKernel = program->createKernel("lineSearchStep");
    lineSearchStepKernel->addArg(x);
    lineSearchStepKernel->addArg(xPrev);
    lineSearchStepKernel->addArg(grad);
    lineSearchStepKernel->addArg(gradPrev);
    lineSearchStepKernel->addArg(dir);
    lineSearchStepKernel->addArg(returnFlag);
    lineSearchStepKernel->addArg(gradNorm);
    lineSearchStepKernel->addArg(lineSearchData);
    lineSearchStepKernel->addArg(lineSearchDataBackup);
    lineSearchStepKernel->addArg(reductionPartials1);
    lineSearchStepKernel->addArg(numVariables);

    lineSearchDotKernel = program->createKernel("lineSearchDot");
    lineSearchDotKernel->addArg(grad);
    lineSearchDotKernel->addArg(dir);
    lineSearchDotKernel->addArg(lineSearchData);
    lineSearchDotKernel->addArg(returnFlag);
    lineSearchDotKernel->addArg(returnValue);
    lineSearchDotKernel->addArg(reductionPartials1);
    lineSearchDotKernel->addArg(numVariables);
    lineSearchDotKernel->addArg(); // energy

    lineSearchContinueKernel = program->createKernel("lineSearchContinue");
    lineSearchContinueKernel->addArg(returnFlag);
    lineSearchContinueKernel->addArg(gradNorm);
    lineSearchContinueKernel->addArg(lineSearchData);

    // Generic single-block finalizer that sums a partials buffer into the
    // selected slot of a destination ComputeArray.  Both the source partials
    // and destination array are re-bound per call site via setArg().
    finalizeReductionKernel = program->createKernel("finalizeReduction");
    finalizeReductionKernel->addArg(); // partials
    finalizeReductionKernel->addArg(); // dest
    finalizeReductionKernel->addArg(); // destOffset
    finalizeReductionKernel->addArg(maxReductionBlocks);

    downloadStartEvent = cc.createEvent();
    downloadFinishEvent = cc.createEvent();
    downloadQueue = cc.createQueue();

    // Upload constraint data.

    if (numConstraints) {
        constraintIndices.upload(hostConstraintIndices);
        constraintDistances.upload(hostConstraintDistances, true);
    }
}

void CommonMinimizeKernel::lbfgs(ContextImpl& context) {
    // Evaluate the energy and gradient at the starting point.

    evaluateGpu(context);
    energy += downloadReturnValueSync();
    if (!(fabs(energy) < std::numeric_limits<float>::max())) {
        energy = evaluateCpu(context);
    }
    if (!isfinite(energy)) {
        throw OpenMMException("Energy or force at minimization starting point is infinite or NaN.");
    }

    // Check to see if the starting point is already a minimum.  Since we are at
    // the start of the run, just use the fallback single-thread-block kernel.

    gradNormKernel->execute(threadBlockSize, threadBlockSize);
    if (downloadGradNormSync() <= tolerance) {
        return;
    }

    initializeDirKernel->execute(numVariables);
    for (int iteration = 1, end = 0;;) {
        // Prepare for a line search.

        energyStart = energy;
        // lineSearchSetup reduces grad.dir into lineSearchData[LS_DOT_START].
        cc.clearBuffer(reductionPartials1);
        lineSearchSetupKernel->execute(numVariables);
        finalizeReduction(reductionPartials1, lineSearchData, 0 /* LS_DOT_START */);

        // Take line search steps.

        for (int count = 0;; count++) {
            // lineSearchStep reduces |grad|^2 into gradNorm in the LS_SUCCEED
            // branch (and writes nothing to partials in the other branches).
            cc.clearBuffer(reductionPartials1);
            lineSearchStepKernel->execute(numVariables);
            finalizeReduction(reductionPartials1, gradNorm, 0);

            if (count) {
                int hostReturnFlag = downloadReturnFlagFinish();
                if (hostReturnFlag == 1) {
                    break; // Line search success
                }
                else if (hostReturnFlag == 0 || count >= maxLineSearchIterations) {
                    xPrev.copyTo(x);
                    gradPrev.copyTo(grad);
                    return; // Line search failure
                }
            }

            // Evaluate the energy and gradient at the new search point, then
            // decide if and how to continue the line search.

            evaluateGpu(context);
            downloadReturnValueStart();
            runLineSearchKernels();

            energy += downloadReturnValueFinish();
            if (!(fabs(energy) < std::numeric_limits<float>::max())) {
                // Overflow on the GPU: try the CPU.

                energy = evaluateCpu(context);

                // We ran the line search kernels with an invalid gradient, so
                // we need to reset the state to before they ran and retry.

                lineSearchDataBackup.copyTo(lineSearchData);

                int hostReturnFlag = 2; // Continue the line search
                returnFlag.upload(&hostReturnFlag);

                // lineSearchDot will try to read any restraint energy from
                // returnValue, but it will be FLT_MAX from the failed GPU run,
                // so reset it to 0 (since the restraint energy from the CPU run
                // is included in the return value that will get uploaded).

                cc.clearBuffer(returnValue);

                runLineSearchKernels();
            }


            downloadReturnFlagStart();
        }

        // Check for convergence or exceeding the maximum number of steps.

        if ((reporter != NULL && report(context, iteration)) || (maxIterations && maxIterations < iteration + 1)) {
            return;
        }

        // Do L-BFGS update of search direction.

        if (largeGrad) {
            gradNormKernel->execute(threadBlockSize, threadBlockSize);
        }
        getDiffKernel->setArg(12, end);
        getDiffKernel->setArg(13, (int) largeGrad);
        getDiffKernel->execute(numVariables);

        downloadReturnFlagStart();

        getScaleKernel->setArg(9, end);
        getScaleKernel->setArg(10, (int) largeGrad);
        if (largeGrad) {
            // Fallback single-block path writes scale[end] and returnValue
            // directly; no finalize needed.
            getScaleKernel->execute(threadBlockSize, threadBlockSize);
        }
        else {
            // Multi-block path emits two partial-sum buffers (xGrad and
            // gradGrad).  Each must be reduced deterministically.
            cc.clearBuffer(reductionPartials1);
            cc.clearBuffer(reductionPartials2);
            getScaleKernel->execute(numVariables);
            finalizeReduction(reductionPartials1, scale, end);
            finalizeReduction(reductionPartials2, returnValue, 0);
        }

        int limit = min(numVectors, iteration++);
        int vectorIndex = end;
        if (++end >= numVectors) {
            end -= numVectors;
        }

        reinitializeDirKernel->setArg(9, vectorIndex);
        reinitializeDirKernel->setArg(10, (int) largeGrad);
        cc.clearBuffer(reductionPartials1);
        reinitializeDirKernel->execute(numVariables);
        finalizeReduction(reductionPartials1, alpha, vectorIndex);

        for (int vector = 0; vector < limit; vector++) {
            if (vector && --vectorIndex < 0) {
                vectorIndex += numVectors;
            }

            if (vector < limit - 1) {
                updateDirAlphaKernel->setArg(8, vectorIndex);
                // Destination index inside the kernel:
                //   vectorIndex2 = (vectorIndex ? vectorIndex : numVectors) - 1
                int destIndex = (vectorIndex ? vectorIndex : numVectors) - 1;
                cc.clearBuffer(reductionPartials1);
                updateDirAlphaKernel->execute(numVariables);
                finalizeReduction(reductionPartials1, alpha, destIndex);
            }
        }

        scaleDirKernel->setArg(8, vectorIndex);
        cc.clearBuffer(reductionPartials1);
        scaleDirKernel->execute(numVariables);
        // scaleDir reduces into the extra alpha[numVectors] slot.
        finalizeReduction(reductionPartials1, alpha, numVectors);

        for (int vector = 0; vector < limit - 1; vector++) {
            // scaleDirKernel puts its first result in alpha[numVectors], so for the
            // first vector, load the result from here instead of alpha[vectorIndex]

            updateDirBetaKernel->setArg(8, vectorIndex);
            updateDirBetaKernel->setArg(9, vector ? vectorIndex : numVectors);
            // Destination index inside the kernel:
            //   vectorIndex2 = (vectorIndex == numVectors-1 ? 0 : vectorIndex+1)
            int destIndex = (vectorIndex == numVectors - 1 ? 0 : vectorIndex + 1);
            cc.clearBuffer(reductionPartials1);
            updateDirBetaKernel->execute(numVariables);
            finalizeReduction(reductionPartials1, alpha, destIndex);

            if (++vectorIndex >= numVectors) {
                vectorIndex -= numVectors;
            }
        }

        // If this is the first iteration, limit is 1, we did not go through the loop above,
        // and we need to read the output of scaleDirKernel directly from alpha[numVectors]

        updateDirFinalKernel->setArg(6, vectorIndex);
        updateDirFinalKernel->setArg(7, limit > 1 ? vectorIndex : numVectors);
        updateDirFinalKernel->execute(numVariables);

        if (downloadReturnFlagFinish()) {
            return;
        }
    }
}

void CommonMinimizeKernel::evaluateGpu(ContextImpl& context) {
    largeGrad = false;

    // Put the current positions in posq and compute virtual site positions.

    restorePosKernel->setArg(2, x);
    restorePosKernel->execute(numParticles);
    context.computeVirtualSites();

    // Evaluate the forces and energy for the desired interactions as well as
    // harmonic restraints to emulate the constraints.

    {
        ContextDeselector deselector(cc);
        energy = context.calcForcesAndEnergy(true, true, forceGroups);
    }

    if (numConstraints) {
        if (mixedIsDouble) {
            getConstraintEnergyForcesKernel->setArg(8, kRestraint);
        }
        else {
            getConstraintEnergyForcesKernel->setArg(8, (float) kRestraint);
        }
        // restorePos has already cleared returnValue to 0 (see restorePos in
        // minimize.cc).  Clear the partials buffer so blocks past the launched
        // grid contribute 0, then reduce the per-block restraint energies into
        // returnValue deterministically.
        cc.clearBuffer(reductionPartials1);
        getConstraintEnergyForcesKernel->execute(numConstraints);
        finalizeReduction(reductionPartials1, returnValue, 0);
    }

    // Convert the forces from fixed to floating point format.  If they are too
    // large, the energy in returnValue will be set to FLT_MAX to signal that the
    // results are invalid and we must fall back to computing forces on the CPU.

    convertForcesKernel->execute(numParticles);
}

double CommonMinimizeKernel::evaluateCpu(ContextImpl& context) {
    largeGrad = true;

    // Create a CPU context if one has not already been created.

    const System& system = context.getSystem();

    if (cpuContext == NULL) {
        Platform* cpuPlatform;
        try {
            cpuPlatform = &Platform::getPlatformByName("CPU");
        }
        catch (...) {
            cpuPlatform = &Platform::getPlatformByName("Reference");
        }
        cpuContext = new Context(system, cpuIntegrator, *cpuPlatform);
    }

    // Download positions and evaluate forces on the CPU.

    x.download(hostX, true);
    for (int i = 0; i < numParticles; i++) {
        hostPositions[i] = Vec3(hostX[3 * i], hostX[3 * i + 1], hostX[3 * i + 2]);
    }
    cpuContext->setState(context.getOwner().getState(State::Parameters));
    cpuContext->setPositions(hostPositions);
    cpuContext->computeVirtualSites();
    State state;
    {
        ContextDeselector deselector(cc);
        state = cpuContext->getState(State::Energy | State::Forces, false, forceGroups);
    }
    double hostEnergy = state.getPotentialEnergy();
    const vector<Vec3>& hostForces = state.getForces();

    // Prepare the gradient to send back to the optimizer.

    for (int i = 0; i < numParticles; i++) {
        if (system.getParticleMass(i) != 0) {
            hostGrad[3 * i] = -hostForces[i][0];
            hostGrad[3 * i + 1] = -hostForces[i][1];
            hostGrad[3 * i + 2] = -hostForces[i][2];
        }
    }

    // Apply harmonic forces for constraints.

    for (int i = 0; i < numConstraints; i++) {
        mm_int2 indices = hostConstraintIndices[i];
        double distance = hostConstraintDistances[i];
        Vec3 delta = hostPositions[indices.y] - hostPositions[indices.x];
        double r2 = delta.dot(delta);
        double r = sqrt(r2);
        delta *= 1 / r;
        double dr = r - distance;
        double kdr = kRestraint * dr;
        hostEnergy += 0.5 * kdr * dr;
        if (system.getParticleMass(indices.x) != 0) {
            hostGrad[3 * indices.y] -= kdr * delta[0];
            hostGrad[3 * indices.y + 1] -= kdr * delta[1];
            hostGrad[3 * indices.y + 2] -= kdr * delta[2];
        }
        if (system.getParticleMass(indices.y) != 0) {
            hostGrad[3 * indices.x] += kdr * delta[0];
            hostGrad[3 * indices.x + 1] += kdr * delta[1];
            hostGrad[3 * indices.x + 2] += kdr * delta[2];
        }
    }

    // Check for overflow of the forces.  We need to check for this here and
    // either back up the line search or abort, since the GPU platforms won't
    // check for NaN positions, and the optimizer could get stuck otherwise.

    if (mixedIsDouble) {
        for (int i = 0; i < numVariables; i++) {
            if (!isfinite(hostGrad[i])) {
                return NAN;
            }
        }
    }
    else {
        for (int i = 0; i < numVariables; i++) {
            if (!isfinite((float) hostGrad[i])) {
                return NAN;
            }
        }
    }

    // Upload forces and return the final energy.

    grad.upload(hostGrad, true);
    return hostEnergy;
}

bool CommonMinimizeKernel::report(ContextImpl& context, int iteration) {
    x.download(hostX, true);
    grad.download(hostGrad, true);

    double restraintEnergy = 0.0, maxError = 0.0;
    for (int i = 0; i < numConstraints; i++) {
        mm_int2 indices = hostConstraintIndices[i];
        double distance = hostConstraintDistances[i];
        Vec3 delta = Vec3(hostX[3 * indices.y] - hostX[3 * indices.x], hostX[3 * indices.y + 1] - hostX[3 * indices.x + 1], hostX[3 * indices.y + 2] - hostX[3 * indices.x + 2]);
        double r2 = delta.dot(delta);
        double r = sqrt(r2);
        double dr = r - distance;
        restraintEnergy += 0.5 * kRestraint * dr * dr;
        maxError = max(maxError, fabs(dr) / distance);
    }

    map<string, double> args;
    args["restraint energy"] = restraintEnergy;
    args["system energy"] = energy - restraintEnergy;
    args["restraint strength"] = kRestraint;
    args["max constraint error"] = maxError;
    {
        ContextDeselector deselector(cc);
        return reporter->report(iteration - 1, hostX, hostGrad, args);
    }
}

void CommonMinimizeKernel::downloadReturnFlagStart() {
    downloadStartEvent->enqueue();
    cc.setCurrentQueue(downloadQueue);
    downloadStartEvent->queueWait(downloadQueue);
    returnFlag.download(pinnedMemory, false);
    downloadFinishEvent->enqueue();
    cc.restoreDefaultQueue();
}

void CommonMinimizeKernel::downloadReturnValueStart() {
    downloadStartEvent->enqueue();
    cc.setCurrentQueue(downloadQueue);
    downloadStartEvent->queueWait(downloadQueue);
    returnValue.download(pinnedMemory, false);
    downloadFinishEvent->enqueue();
    cc.restoreDefaultQueue();
}

int CommonMinimizeKernel::downloadReturnFlagFinish() {
    downloadFinishEvent->wait();
    return *(int*) pinnedMemory;
}

double CommonMinimizeKernel::downloadReturnValueFinish() {
    downloadFinishEvent->wait();
    if (mixedIsDouble) {
        return *(double*) pinnedMemory;
    }
    else {
        return (double) *(float*) pinnedMemory;
    }
}

double CommonMinimizeKernel::downloadReturnValueSync() {
    if (mixedIsDouble) {
        double hostReturnValue;
        returnValue.download(&hostReturnValue);
        return hostReturnValue;
    }
    else {
        float hostReturnValue;
        returnValue.download(&hostReturnValue);
        return (double) hostReturnValue;
    }
}

double CommonMinimizeKernel::downloadGradNormSync() {
    if (mixedIsDouble) {
        double hostGradNorm;
        gradNorm.download(&hostGradNorm);
        return hostGradNorm;
    }
    else {
        float hostGradNorm;
        gradNorm.download(&hostGradNorm);
        return (double) hostGradNorm;
    }
}

void CommonMinimizeKernel::runLineSearchKernels() {
    if (mixedIsDouble) {
        lineSearchDotKernel->setArg(7, isfinite(energy) ? energy - energyStart : (double) std::numeric_limits<float>::max());
    }
    else {
        lineSearchDotKernel->setArg(7, isfinite((float) energy) ? (float) (energy - energyStart) : std::numeric_limits<float>::max());
    }
    // lineSearchDot reduces grad . dir into lineSearchData[LS_DOT] via the
    // deterministic two-pass path.  Clear partials first so blocks that early
    // return on LS_FAIL/LS_SUCCEED contribute zero.
    cc.clearBuffer(reductionPartials1);
    lineSearchDotKernel->execute(numVariables);
    finalizeReduction(reductionPartials1, lineSearchData, 1 /* LS_DOT */);
    lineSearchContinueKernel->execute(1);
}

void CommonMinimizeKernel::finalizeReduction(ComputeArray& partials, ComputeArray& dest, int destOffset) {
    finalizeReductionKernel->setArg(0, partials);
    finalizeReductionKernel->setArg(1, dest);
    finalizeReductionKernel->setArg(2, destOffset);
    // numPartials at arg 3 is bound once to maxReductionBlocks in setup().
    finalizeReductionKernel->execute(threadBlockSize, threadBlockSize);
}
