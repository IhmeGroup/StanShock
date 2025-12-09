from __future__ import annotations

import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.physics.fluid_base import FluidState
from stanshock.processing.probe import Probe
from stanshock.system.backend import Array
from stanshock.system.geometry import ConstantValue, Cylinder


class ShockTube(Combustor):
    """
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver.
    """

    def pressure_rise(self, t, p, peakWidth=10):
        """
        This method attempts to determine the pressure rise based on the separation
        of the first two peaks in the logarithmic derivative of the of the
        endwall pressure. This method is the most robust when only the incident
        shock provides the only peak in the logarithmic derivative of pressure.
            inputs:
                t = time [s]
                p = pressure [pa]
                peakWidth (optional) =  # of samples to define a peak; this is
                                        also the number of points used to define
                                        the pressure rise region.
            output:
                dlnpdt: mean logarithmic slope in the pressure rise region.
                p5: the mean test pressure
        """

        from scipy import signal

        # find the logarithmic time-derivative of pressure
        lnp = np.log(p)
        dlnpdt = np.diff(lnp) / np.diff(t)
        # use a toolbox to find the peaks of the order of the peakWidth
        peakIndices = signal.find_peaks_cwt(dlnpdt, np.array([peakWidth]))
        peakIndices = np.append(peakIndices, len(t) - 2)  # add final point
        # assume the top peak is the incident shock and remove peaks that come before
        peakIndices = peakIndices[np.argsort(-np.abs(dlnpdt[peakIndices]))]
        peakIndices = [
            peakIndex for peakIndex in peakIndices if peakIndex >= peakIndices[0]
        ]
        lowerIndex = int(float(peakIndices[0] + peakIndices[1] - peakWidth) / 2)
        upperIndex = int(float(peakIndices[0] + peakIndices[1] + peakWidth) / 2)
        return np.mean(dlnpdt[lowerIndex:upperIndex]), np.mean(p[lowerIndex:upperIndex])

    def optimize_driver_insert(
        self, tFinal, tradeoffParam=1.0, tTest=None, p5=None, eps=1e-4, maxIter=100
    ):
        """
        This method finds the driver insert geometry, which minimizes the
        pressure rise due to boundary layer effects while obtaining the test
        pressure. The final state of this function is the optimized state.
            inputs
                    tFinal = final time
                    tTest = the test time. This is used for normalization in the
                        cost function.
                    tradeoffParam = emphasis for experiment. A higher value
                        indicates that a correct test pressure is more valuable.
                    p5 = test pressure
                    eps = cutoff parameter for the global search. A higher value
                        indicates a tighter tolerance.
                    maxIter = maximum number of iterations
        """
        from scipy.optimize import newton
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF,
        )  # RBF is the gaussian correlation

        # Check for boundary layer terms
        if not hasattr(self, "boundary_layer"):
            msg = "Boundary layer terms have not been set."
            raise AttributeError(msg)

        # Initialize the state
        self.state = self.initialization()
        self.state.gamma = self.physics.get_gamma(self.state)

        assert isinstance(self.geometry, Cylinder)
        geometry: Cylinder = self.geometry
        msg = None
        if geometry.d_outer is None or geometry.dlnA_dx is None:
            msg = "Driver optimization must have d_outer and dlnA_dx defined"
        if (
            not isinstance(geometry.d_inner, ConstantValue)
            or geometry.d_inner.constant != 0
        ):
            msg = "Driver optimization cannot have an inner diameter"
        assert self.state.pressure is not None
        if self.state.pressure[0] < self.state.pressure[-1]:
            msg = "Optimization routine requires the driver gas to be on the left."
        if msg is not None:
            raise Exception(msg)

        if tTest is None:
            if self.verbose:
                print(
                    "WARNING: the normalization time is not included. Setting to the simulation time."
                )
            tTest = tFinal
        if p5 is None and tradeoffParam != 0:
            if self.verbose:
                print(
                    "WARNING: the target test pressure is not provided. Determining p5 via normal shock relations"
                )
            g1, g4 = self.state.gamma[-1], self.state.gamma[0]
            p4op1 = self.state.pressure[0] / self.state.pressure[-1]
            r4or1 = self.state.density[0] / self.state.density[-1]
            a4oa1 = np.sqrt(g4 / g1 * p4op1 / r4or1)

            def res(Ms1):
                p4op1 - (1.0 + 2.0 * g1 / (g1 + 1.0) * (Ms1**2.0 - 1.0)) * (
                    1.0 - (g4 - 1.0) / (g4 + 1.0) / a4oa1 * (Ms1 - 1.0 / Ms1)
                ) ** (-2.0 * g4 / (g4 - 1.0))

            Ms1 = newton(res, 2.0)
            p5op1 = ((2.0 * g1 * Ms1**2.0 - (g1 - 1.0)) / (g1 + 1.0)) * (
                (-2.0 * (g1 - 1.0) + Ms1**2.0 * (3.0 * g1 - 1.0))
                / (2.0 + Ms1**2.0 * (g1 - 1.0))
            )
            p5 = p5op1 * self.state.pressure[-1]
        # Get initial state for reinitialization
        idx = self.geometry.idx_cells
        rInitial = np.copy(self.state.density[idx])
        uInitial = np.copy(self.state.velocity[idx])
        pInitial = np.copy(self.state.pressure[idx])
        YInitial = np.copy(self.state.composition[idx])
        gammaInitial = np.copy(self.state.gamma[idx])
        dlnA_dx_initial = geometry.dlnA_dx

        def dd_outerdx(time: float, x: Array) -> Array:
            return (
                geometry.d_outer(time, x) / 2.0 * dlnA_dx_initial(0.0, x)
            )  # assume temporally constant area

        # Determine geometry from pressure
        dpAbs = np.abs(np.diff(pInitial))
        xShock = geometry.xf[
            np.argmax(dpAbs) + 1
        ]  # maximum pressure gradient corresponds to shock
        (xMin, xMax, probeLocation) = (geometry.xf[0], xShock, geometry.xf[-1])
        LMax = xMax - xMin  # maximum length of constrained optimization
        DMax = min(
            geometry.d_outer(0.0, np.linspace(xMin, xMax))
        )  # maximum diameter of constrained optimization
        smoothingLength = 10 * geometry.dx
        if LMax <= smoothingLength:
            msg = "This calculation will likely be unstable. Refine the grid"
            raise Exception(msg)

        def alphaMax(L):
            return 1.0 - smoothingLength / L

        (LMin, DMin, alphaMin) = (
            smoothingLength,
            0.0,
            0.0,
        )  # no driver bound (no negative lengths)

        # calculate a smoothing length for numerical stability
        #######################################################################
        def optimization_function(design):
            """
            This function solves the shocktube problem and returns the absolute pressure rise. The insert geometry is assumed to
            vary linearly in area
                inputs:
                    design=tuple with
                        (total length of insert, maximum diameter of insert, ratio of constant portion of insert to entire insert)

            """
            # determine the geometry
            (LInsert, DInsert, alpha) = design
            xIns0, xIns1 = xMin + LInsert * alpha, xMin + LInsert
            AIns0 = np.pi * DInsert**2.0 / 4.0

            def AInsert(time: float, x: Array) -> Array:
                AIns = np.zeros_like(x)
                inds = np.logical_and(x >= xIns0, x < xIns1)
                AIns[inds] = AIns0 * (1.0 - (x[inds] - xIns0) / (xIns1 - xIns0))
                AIns[x < xIns0] = AIns0
                return AIns

            def dAInsertdx(time: float, x: Array) -> Array:
                dAInsdx = np.zeros_like(x)
                inds = np.logical_and(x >= xIns0, x < xIns1)
                dAInsdx[inds] = -AIns0 / (xIns1 - xIns0)
                return dAInsdx

            def d_inner(time: float, x: Array) -> Array:
                return np.sqrt(4.0 * AInsert(time, x) / np.pi)

            def dd_innerdx(time: float, x: Array) -> Array:
                dDIndx = np.zeros_like(x)
                inds = np.logical_and(x >= xIns0, x < xIns1)
                dDIndx[inds] = (
                    0.5
                    * (4.0 * AInsert(time, x[inds]) / np.pi) ** -0.5
                    * (4.0 * dAInsertdx(time, x[inds]) / np.pi)
                )
                return dDIndx

            def A(time: float, x: Array) -> Array:
                return (
                    0.25
                    * np.pi
                    * (geometry.d_outer(time, x) ** 2.0 - d_inner(time, x) ** 2.0)
                )

            def dA_dx(time: float, x: Array) -> Array:
                return (
                    0.5
                    * np.pi
                    * (
                        geometry.d_outer(time, x) * dd_outerdx(time, x)
                        - d_inner(time, x) * dd_innerdx(time, x)
                    )
                )

            # initialize (may be at a previous state in the optimization)
            geometry.dlnA_dx = lambda t, x: dA_dx(t, x) / A(t, x)
            geometry.d_inner = d_inner
            self.state = FluidState(
                shape=(geometry.n_cells_interior,),
                density=np.copy(rInitial),
                velocity=np.copy(uInitial),
                pressure=np.copy(pInitial),
                composition=np.copy(YInitial),
                gamma=np.copy(gammaInitial),
            )
            self.state = self.inviscid_flux.face_extrapolator.add_ghost_layers(
                self.state
            )
            # delete previous probes and create an endwall probe
            self.probes = [
                Probe(self, probeLocation, skipSteps=0, probeName="endwall probe"),
            ]
            # solve
            if self.verbose:
                print(
                    f"Solving Optimization. Iteration={self.optimization_iteration}, L={LInsert:.3f}, D={DInsert:.3f}, alpha={alpha:.3f}"
                )
            self.t = 0.0
            self.advance_simulation(tFinal)
            self.optimization_iteration += 1
            # return
            dlnpdt, p5Act = self.pressure_rise(
                np.array(self.probes[0].t), np.array(self.probes[0].p)
            )
            if self.verbose:
                print(
                    f"Finished with optimization iteration. dlnpdt={dlnpdt:f}, p5={p5Act:f}"
                )
            return (dlnpdt * tTest) ** 2.0 + tradeoffParam * (p5Act / p5 - 1.0) ** 2.0

        #######################################################################
        def midpoint_vector(xMin, xMax, nMidpoints):
            """
            This function returns a vector to sample from. The vector is
            uniformly space and is non-inclusive of the bounds
                inputs:
                    xMin=lower bound
                    xMax=upper bound
                    nMidpoints=number of sampling locations
                output:
                    sample vector
                    spacing (dx)
            """
            dx = (xMax - xMin) / float(nMidpoints)
            return xMin + dx / 2.0 + dx * np.arange(0, nMidpoints)

        #######################################################################
        # develop initial grid of points
        nGrid = 3
        self.designs = []
        for L in midpoint_vector(LMin, LMax, nGrid):
            for D in midpoint_vector(DMin, DMax, nGrid):
                for a in midpoint_vector(alphaMin, alphaMax(L), nGrid):
                    self.designs.append((L, D, a))
        # solve for each grid point on the initial parameter space
        nDesigns = len(self.designs)
        self.yOpt = []  # evaluated points
        if self.verbose:
            print(f"Solving initial grid of {nDesigns} points.")
        for _iDesign, design in enumerate(self.designs):
            self.yOpt.append(optimization_function(design))
        # initialize the Gaussian Random Process as a surrogate
        kernel = 1.0 * RBF(
            length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
        )  # initialize
        gp = GaussianProcessRegressor(kernel=kernel)
        # determine the grid to search over with the surrogate model
        nHat = 30
        XHat = []
        for L in midpoint_vector(LMin, LMax, nHat):
            for D in midpoint_vector(DMin, DMax, nHat):
                for a in midpoint_vector(alphaMin, alphaMax(L), nHat):
                    XHat.append((L, D, a))
        XHat = np.array(XHat)
        # iterate until optimum is found
        ymin = min(self.yOpt)
        if self.verbose:
            print("Finding Optimum.")
        minImprovement = eps / 10.0
        maxImprovement = minImprovement + 1.0
        while (
            ymin > eps
            and self.optimization_iteration < maxIter
            and maxImprovement > minImprovement
        ):
            # fit the GP
            X = np.array(self.designs)
            gp.fit(X, np.array(self.yOpt))
            YHat = gp.predict(XHat)
            # find the improvement
            parameters = gp.kernel_.get_params()
            sigmaSqrd = parameters["k1__constant_value"]
            R = gp.kernel_.k2  # correlation matrix
            RInv = np.linalg.inv(R(X))
            ExpImprovements = np.zeros(XHat.shape[0])
            ymin = min(self.yOpt)
            rs = R(X, Y=XHat)
            r = np.sum(RInv.dot(rs) * rs, axis=0)
            sSqrds = sigmaSqrd * (1.0 - r)
            ind = sSqrds <= 0
            ExpImprovements[ind] = np.maximum(
                ymin - YHat[ind], np.zeros_like(YHat[ind])
            )
            ind = sSqrds > 0
            s = np.sqrt(sSqrds[ind])
            T = (ymin - YHat[ind]) / s
            ExpImprovements[ind] = s * (T * norm.cdf(T) + norm.pdf(T))
            # find the maximum improvement and compute the new datum
            maxImprovement = np.max(ExpImprovements)
            self.designs.append(XHat[np.argmax(ExpImprovements), :])
            self.yOpt.append(optimization_function(self.designs[-1]))
            if self.verbose:
                print(
                    f"Minimum of current iteration: {ymin:f}. Expected improvement of the next iteration: {maxImprovement:f}"
                )
        if self.optimization_iteration >= maxIter and self.verbose:
            print("No minimum found within tolerance.")
        elif maxImprovement <= minImprovement and self.verbose:
            print(
                "Search stopped due to no sample points found yielding enough improvement."
            )
        elif self.verbose:
            print("Minimum Found. Setting to minimum state.")
        optimization_function(self.designs[np.argmin(self.yOpt)])
