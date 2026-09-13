import cvxpy as cp
import numpy as np
from scipy.special import kl_div
from functions import *
import os
import warnings


class CES:
    def __init__(self, v, rho, B=None, sparse=False):
        v = np.asarray(v, dtype=float)
        if v.ndim != 2 or 0 in v.shape:
            raise ValueError("v must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("v must contain finite, nonnegative valuations")
        if np.any(~np.any(v > 0, axis=1)):
            raise ValueError("every buyer must value at least one good")
        if not np.isscalar(rho):
            raise ValueError("rho must be a finite scalar satisfying 0 < rho < 1")
        try:
            rho = float(rho)
        except (TypeError, ValueError):
            raise ValueError("rho must be a finite scalar satisfying 0 < rho < 1") from None
        if not np.isfinite(rho) or not 0 < rho < 1:
            raise ValueError("rho must be a finite scalar satisfying 0 < rho < 1")

        self.n, self.m = v.shape
        self.rho = rho
        self.index_list = np.arange(1, self.n + 1)
        self.index_matrix = np.arange(1, self.n + 1).reshape(-1, 1) * np.ones(self.m)
        if B is None:
            self.B = np.ones(shape=self.n)
        else:
            self.B = np.asarray(B, dtype=float)
            if self.B.shape != (self.n,):
                raise ValueError(f"B must have shape ({self.n},)")
            if not np.all(np.isfinite(self.B)) or np.any(self.B <= 0):
                raise ValueError("B must contain finite, strictly positive budgets")
        self.v = v.copy()

        self.sparse = sparse
        self.v_one_cols = np.sum(self.v != 0, axis=0) != 1
        self.v_nonzero = (self.v != 0)

        self.x = (self.B / np.sum(self.B)).reshape(-1, 1) * np.ones((self.n, self.m))
        self.x_ = self.x.copy()
        self.u_rho = np.sum(self.v * self.x ** self.rho, axis=1)
        self.u_rho_min = self.u_rho.copy()
        delta_1 = self.B / (self.m * sum(self.B))
        # The lower bound uses the least *positive* valuation.  Zero entries in
        # a sparse market represent missing edges and are not part of delta_2.
        min_v = np.min(np.where(self.v > 0, self.v, np.inf), axis=1)
        max_v = np.amax(self.v, axis=1)
        delta_2 = min_v / max_v
        self.x_min = (delta_1 ** (1 / (1 - self.rho)) * delta_2 ** (
                (self.rho + 1) / (1 - self.rho))).reshape(-1, 1) \
                     * np.ones(self.m)

        if not np.all(np.isfinite(self.x_min)) or np.any(self.x_min <= 0):
            raise ValueError("the CES smoothing bound is outside floating-point range")

        self.cons_2_x = self.rho * (self.rho - 1) / 2 * self.v * self.x_min ** (self.rho - 2)
        self.cons_1_x = self.rho * (2 - self.rho) * self.v * self.x_min ** (self.rho - 1)
        self.cons_0_x = (1 - self.rho / 2) * (1 - self.rho) * self.v * self.x_min ** self.rho

        self.cons_2 = - self.B / (2 * self.rho * self.u_rho_min ** 2)
        self.cons_1 = 2 * self.B / (self.rho * self.u_rho_min)
        self.cons_0 = self.B / self.rho * np.log(self.u_rho_min) - 3 / 2 * self.B / self.rho

        self.b = (self.B / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        self.p = (np.sum(self.B) / self.m) * np.ones(self.m)
        self.p_ = self.p.copy()
        self.beta = self.B / self.u_rho ** (1 / self.rho)
        self.opt_x = self.x.copy()
        self.opt_u = self.u_rho ** (1 / self.rho)
        self.opt_p = self.p.copy()
        self.opt_b = self.b.copy()
        self.file = None

    def initialize(self, alpha=0.06):
        if not np.isscalar(alpha):
            raise ValueError("alpha must be a finite, strictly positive scalar")
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            raise ValueError("alpha must be a finite, strictly positive scalar") from None
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite, strictly positive scalar")
        if not self.sparse:
            self.x = (self.B / np.sum(self.B)).reshape(-1, 1) * np.ones((self.n, self.m))
            self.x_ = self.x.copy()
            self.u_rho = np.sum(self.v * self.x ** self.rho, axis=1)
            self.p = (np.sum(self.B) / self.m) * np.ones(self.m)
            self.p_ = self.p.copy()
            self.beta = self.B / self.u_rho ** (1 / self.rho)
            self.b = (self.B / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        else:
            self.solve_pr(num_iter=1, record=False, processing=False, init=True)
            self.x_ = self.x.copy()
            self.p_ = self.p.copy()
            self.u_rho_min = alpha * self.u_rho
            self.cons_2 = - self.B / (2 * self.rho * self.u_rho_min ** 2)
            self.cons_1 = 2 * self.B / (self.rho * self.u_rho_min)
            self.cons_0 = self.B / self.rho * np.log(self.u_rho_min) - 3 / 2 * self.B / self.rho

    def phi(self):
        u_rho = self._raw_u_rho(self.x)

        return np.sum(self.B / self.rho * np.log(u_rho))

    def _raw_u_rho(self, allocation):
        return np.sum(self.v * np.power(allocation, self.rho), axis=1)

    def _supporting_prices(self, beta, allocation):
        """Return finite prices defining a supporting hyperplane for each CES utility.

        CES marginal utilities are infinite at a valued zero coordinate.  In
        that case any strictly positive reference bundle gives a valid
        supporting hyperplane, so use the already-computed smoothing lower
        bound as the reference point.  This keeps diagnostic prices finite
        without changing the allocation or the true primal objective.
        """
        reference_x = np.asarray(allocation, dtype=float).copy()
        valued = self.v > 0
        reference_x[valued] = np.maximum(reference_x[valued], self.x_min[valued])
        reference_u = self._raw_u_rho(reference_x) ** (1 / self.rho)

        weighted_power = np.zeros_like(reference_x)
        weighted_power[valued] = (
            self.v[valued] * np.power(reference_x[valued], self.rho - 1)
        )
        marginal_u = reference_u.reshape(-1, 1) ** (1 - self.rho) * weighted_power
        return np.max(beta.reshape(-1, 1) * marginal_u, axis=0)

    @staticmethod
    def _solve_problem(problem, solver=None, acceptable_statuses=None):
        """Solve a CVXPY problem with MOSEK when present and open fallbacks otherwise."""
        if acceptable_statuses is None:
            acceptable_statuses = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        if solver is not None:
            candidates = [solver]
        else:
            installed = set(cp.installed_solvers())
            candidates = [name for name in ("MOSEK", "CLARABEL", "SCS") if name in installed]
        if not candidates:
            raise cp.SolverError("no supported conic solver is installed")

        errors = []
        for candidate in candidates:
            try:
                problem.solve(solver=candidate)
            # CVXPY can surface vendor exceptions directly (for example,
            # mosek.Error when the package is installed but unlicensed).
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if problem.status in acceptable_statuses:
                return candidate
            errors.append(f"{candidate}: status={problem.status}")
        raise cp.SolverError("; ".join(errors))

    def dual_phi(self, validate_subproblem=False, solver=None):
        u = self._raw_u_rho(self.x) ** (1 / self.rho)
        beta = self.B / u
        p = self._supporting_prices(beta, self.x)

        # The supporting-gradient construction makes every positively
        # homogeneous utility conjugate equal to zero.  Avoid solving the same
        # conic subproblem at every logging point; callers can request the
        # numerical check explicitly when debugging.
        subproblem_value = self.compute_dual_sub_opt(beta, p, solver=solver) if validate_subproblem else 0.0
        return np.sum(p) + np.sum(self.B * (-1 - np.log(beta) + np.log(self.B))) + subproblem_value

    def compute_dual_sub_opt(self, beta, p, solver=None):
        beta = np.asarray(beta, dtype=float)
        p = np.asarray(p, dtype=float)
        if beta.shape != (self.n,) or p.shape != (self.m,):
            raise ValueError("beta and p must have shapes (n,) and (m,), respectively")
        if (not np.all(np.isfinite(beta)) or not np.all(np.isfinite(p))
                or np.any(beta < 0) or np.any(p < 0)):
            raise ValueError("beta and p must be finite and nonnegative")

        x = cp.Variable((self.n, self.m), nonneg=True)
        ces_utilities = [
            cp.pnorm(cp.multiply(self.v[i] ** (1 / self.rho), x[i]), self.rho)
            for i in range(self.n)
        ]
        term1 = cp.sum(cp.hstack([beta[i] * ces_utilities[i] for i in range(self.n)]))
        prob = cp.Problem(cp.Maximize(term1 - cp.sum(cp.multiply(x, p.reshape(1, -1)))))
        self._solve_problem(
            prob,
            solver=solver,
            acceptable_statuses=(
                cp.OPTIMAL, cp.OPTIMAL_INACCURATE,
                cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE,
            ),
        )
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return prob.value
        if prob.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
            return np.inf
        return np.nan

    def quasi_phi(self, a=None):
        u_rho = self.quasi_u_rho(a=a)
        f = self.cons_0 + self.cons_1 * u_rho + self.cons_2 * u_rho ** 2
        above_bound = u_rho >= self.u_rho_min
        f[above_bound] = self.B[above_bound] / self.rho * np.log(u_rho[above_bound])

        return np.sum(f)

    def _deriv_quasi_outer(self, u_rho):
        derivative = self.cons_1 + 2 * self.cons_2 * u_rho
        above_bound = u_rho >= self.u_rho_min
        derivative[above_bound] = self.B[above_bound] / (self.rho * u_rho[above_bound])
        return derivative

    def deriv_quasi_phi(self):
        u_rho = self.quasi_u_rho()
        return self._deriv_quasi_outer(u_rho).reshape(-1, 1) * self.deriv_quasi_u_rho()

    def _quasi_components(self, allocation):
        components = self.cons_0_x + self.cons_1_x * allocation + self.cons_2_x * allocation ** 2
        above_bound = allocation >= self.x_min
        components[above_bound] = (
            self.v[above_bound] * np.power(allocation[above_bound], self.rho)
        )
        return components

    def _quasi_component_column(self, allocation_column, j):
        component = (self.cons_0_x[:, j] + self.cons_1_x[:, j] * allocation_column
                     + self.cons_2_x[:, j] * allocation_column ** 2)
        above_bound = allocation_column >= self.x_min[:, j]
        component[above_bound] = (
            self.v[above_bound, j] * np.power(allocation_column[above_bound], self.rho)
        )
        return component

    def _deriv_quasi_component_column(self, allocation_column, j):
        derivative = self.cons_1_x[:, j] + 2 * self.cons_2_x[:, j] * allocation_column
        above_bound = (allocation_column >= self.x_min[:, j]) & (self.v[:, j] > 0)
        derivative[above_bound] = (
            self.rho * self.v[above_bound, j]
            * np.power(allocation_column[above_bound], self.rho - 1)
        )
        return derivative

    def quasi_u_rho(self, a=None):
        allocation = self.x if a is None else self.x_
        return np.sum(self._quasi_components(allocation), axis=1)

    def deriv_quasi_u_rho(self):
        derivative = self.cons_1_x + 2 * self.cons_2_x * self.x
        above_bound = (self.x >= self.x_min) & (self.v > 0)
        derivative[above_bound] = (
            self.rho * self.v[above_bound] * np.power(self.x[above_bound], self.rho - 1)
        )
        return derivative

    def dual_gap(self):
        obj_primal = self.phi()
        obj_dual = self.dual_phi()

        return obj_dual - obj_primal

    def utility_gap(self, mode='all'):
        u = self._raw_u_rho(self.x) ** (1 / self.rho)
        u_gap = np.divide(np.absolute(u - self.opt_u), self.opt_u,
                          out=np.full_like(u, np.nan), where=self.opt_u > 0)
        if mode == 'all':
            return u_gap
        elif mode == 'max':
            return np.max(u_gap)
        elif mode == 'avg':
            return np.average(u_gap)
        raise ValueError("mode must be one of 'all', 'max', or 'avg'")

    def price_gap(self, mode='all'):
        u = self._raw_u_rho(self.x) ** (1 / self.rho)
        beta = self.B / u
        p = self._supporting_prices(beta, self.x)
        p_gap = np.divide(np.absolute(p - self.opt_p), self.opt_p,
                          out=np.zeros_like(p), where=self.opt_p > 0)
        p_gap[(self.opt_p <= 0) & (p > 0)] = np.inf
        if mode == 'all':
            return p_gap
        elif mode == 'max':
            return np.max(p_gap)
        elif mode == 'avg':
            return np.average(p_gap)
        raise ValueError("mode must be one of 'all', 'max', or 'avg'")

    def record(self):
        os.makedirs('./records', exist_ok=True)
        self.file = open(f'./records/record-ces-{self.n}-{self.m}', "w", encoding='utf-8')
        self.file.write(f"the number of buyers: {self.n} \n"
                        f"the number of goods: {self.m} \n")
        self.file.write("the valuation matrix: \n")
        for row in self.v:
            self.file.write(str(row) + "\n")
        self.file.write("the budget of buyers: \n" + str(self.B) + "\n\n\n")

    def write(self, content, pos, processing=True, record=True):
        if pos == 'begin':
            if processing:
                print(content)
            if record and self.file is not None:
                self.file.write(content + "\n")
        elif pos == 'end':
            if processing:
                print("------ Done! ------\n")
            if record and self.file is not None:
                self.file.write("\n***{ Solution }***\n" + "allocation: \n")
                for row in self.x:
                    self.file.write(str(np.round(row, 4)) + "\n")
                self.file.write(f"primal objective: {self.phi()}")
                self.file.write("\n\n\n")

    def close_file(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def store_processing_record(self, k, cost, data, store=True, processing=True, record=True, freq=(1, 1, 1)):
        if store:
            if k % max(int(freq[0]), 1) == 0:
                store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())
        if processing:
            if k % freq[1] == 0:
                print(f"--- {k}th iteration finished "
                      f"| dual gap = {data['dual_gap'][-1]} "
                      f"| utility gap = {data['utility_gap(avg)'][-1]} "
                      f"| price gap = {data['price_gap(avg)'][-1]} ---")
        if record:
            pass

    def solve_opt_cvxpy(self, processing=True, record=True, solver=None):
        x = cp.Variable((self.n, self.m), nonneg=True)
        utility_terms = [
            cp.sum(cp.multiply(self.v[i], cp.power(x[i], self.rho)))
            for i in range(self.n)
        ]
        obj = cp.Maximize(cp.sum(cp.hstack([
            self.B[i] / self.rho * cp.log(utility_terms[i]) for i in range(self.n)
        ])))
        constraints = [cp.sum(x, axis=0) <= 1]
        prob = cp.Problem(obj, constraints)
        used_solver = self._solve_problem(prob, solver=solver)
        if processing:
            print(f"> solve primal problem with cvxpy (solver='{used_solver}')")
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            solved_x = np.maximum(np.asarray(x.value, dtype=float), 0)
            self.opt_x = solved_x.copy()
            self.x = solved_x.copy()
            self.x_ = solved_x.copy()
            self.u_rho = self._raw_u_rho(self.x)
            self.opt_u = self.u_rho ** (1 / self.rho)
            dual_prices = constraints[0].dual_value
            if dual_prices is None or not np.all(np.isfinite(dual_prices)):
                self.opt_p = self._supporting_prices(self.B / self.opt_u, self.opt_x)
            else:
                self.opt_p = np.maximum(np.asarray(dual_prices, dtype=float), 0)
            self.opt_b = self.opt_x * self.opt_p.reshape(1, -1)
            if processing:
                print("> SOLVED!")
                print(f"> optimal primal value: {self.phi()}\n")
                print(f"> optimal dual gap: {self.dual_gap()}\n")
            if record and self.file is not None:
                self.file.write(f"> optimal solution: \n{self.x}")
            return prob.value
        raise RuntimeError(f"CVXPY failed to solve the CES reference problem: {prob.status}")

    def solve_bcdeg(self, num_iter, alpha=0.10, eta0=1, factor=(1, 0.80, 1.02), step_size='fixed_step',
                    cyclic=False, processing=True, store=True, record=True, print_=50):
        if step_size not in ('fixed_step', 'line_search', 'adaptive'):
            raise ValueError("step_size must be 'fixed_step', 'line_search', or 'adaptive'")
        self.initialize(alpha=alpha)
        setting = "------ BCDEG ------\n" \
                  + f"- [step size strategy]    \t{step_size}\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        j = -1
        lip_j = np.amax(self.B.reshape(-1, 1) * self.v * self.x_min ** (self.rho - 2) / self.u_rho_min.reshape(-1, 1),
                        axis=0)
        eta = np.divide(eta0, lip_j, out=np.full(self.m, float(eta0)), where=lip_j > 0)
        quasi_u_rho = self.quasi_u_rho()

        if step_size == 'line_search':
            # eta = factor[0] * eta
            eta = factor[0] * np.ones(shape=self.m)

        j_change = True
        for k in range(1, num_iter + 1):
            if j_change:
                if not cyclic:
                    j = np.random.randint(self.m)
                else:
                    j = (j + 1) % self.m

            if step_size == 'adaptive':
                current_component = self._quasi_component_column(self.x[:, j], j)
                u_rho_minus_j = quasi_u_rho - current_component
                # The smoothed outer objective is flat enough below
                # u_rho_min that this is the relevant safe denominator.  The
                # entire expression is buyer-wise and then reduced to the one
                # scalar Lipschitz constant for column j.
                safe_u_rho_minus_j = np.maximum(u_rho_minus_j, self.u_rho_min)
                lip_terms = (self.B * self.v[:, j] * self.x_min[:, j] ** (self.rho - 2)
                             / safe_u_rho_minus_j)
                lip_j = np.max(lip_terms)
                eta[j] = 1 / lip_j if lip_j > 0 else float(eta0)

            update = False

            if self.sparse and not self.v_one_cols[j]:
                cost += self.n
            else:
                current_component = self._quasi_component_column(self.x[:, j], j)
                g = (self._deriv_quasi_outer(quasi_u_rho)
                     * self._deriv_quasi_component_column(self.x[:, j], j))

                d0 = self.x[:, j] + eta[j] * g
                p_j = compute_for_price(d0, eta[j], self.index_list)
                x_j = np.maximum(d0 - eta[j] * p_j, 0)
                candidate_component = self._quasi_component_column(x_j, j)
                quasi_u_rho_ = quasi_u_rho + candidate_component - current_component
                raw_u_rho_ = self.u_rho + self.v[:, j] * (
                    np.power(x_j, self.rho) - np.power(self.x[:, j], self.rho)
                )

                if self.sparse and np.any(self.u_rho_min > raw_u_rho_):
                    warnings.warn("The utility lower bound is not appropriate.")

                cost += self.n

                if step_size == 'line_search':
                    g_j = (self._deriv_quasi_outer(quasi_u_rho_)
                           * self._deriv_quasi_component_column(x_j, j))
                    if np.sum((g_j - g) ** 2) > (1 / eta[j]) ** 2 * np.sum((x_j - self.x[:, j]) ** 2):
                        j_change = False
                        eta[j] = eta[j] * factor[1]
                    else:
                        update = True
                        j_change = True
                        eta[j] = eta[j] * factor[2]

                if update or (step_size == 'fixed_step') or (step_size == 'adaptive'):
                    self.x[:, j] = x_j
                    self.u_rho = raw_u_rho_
                    quasi_u_rho = quasi_u_rho_
                    self.p[j] = p_j

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))

        self.write('', 'end', processing=processing, record=record)
        if store:
            return data

    def solve_pgls(self, num_iter, alpha=0.10, factor=(1e4, 0.80, 1.02), processing=True, store=True,
                   record=True, print_=50):
        self.initialize(alpha=alpha)

        lip = max(self.rho * self.B * np.linalg.norm(self.v, axis=1) ** 2 * np.amin(self.x_min, axis=1) ** (
                2 * self.rho - 2) / self.u_rho_min ** 2)
        eta = factor[0] / lip

        setting = "------ PGLS ------\n" \
                  + f"- [the number of iteration]                             \t{num_iter}\n" \
                  + f"- [the initial stepsize]                                \t{eta} ({factor[0]} / L)\n" \
                  + f"- [the increasing factor in line search]                \t{factor[2]}\n" \
                  + f"- [the decreasing factor in line search]                \t{factor[1]}\n"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        k = 0
        f = -self.quasi_phi()
        g = -self.deriv_quasi_phi()

        update = False
        shrinking_times = 0

        while k < num_iter:
            d0 = self.x - eta * g
            candidate_x = self.x.copy()
            candidate_p = self.p.copy()
            if self.sparse:
                if np.any(self.v_one_cols):
                    d0 = d0[:, self.v_one_cols]
                    candidate_p[self.v_one_cols] = compute_for_price_all(
                        d0, eta, self.index_matrix[:, self.v_one_cols]
                    )
                    candidate_x[:, self.v_one_cols] = np.maximum(
                        d0 - eta * candidate_p[self.v_one_cols], 0
                    )
            else:
                candidate_p = compute_for_price_all(d0, eta, self.index_matrix)
                candidate_x = np.maximum(d0 - eta * candidate_p, 0)
            self.p_ = candidate_p
            self.x_ = candidate_x

            cost += self.m * self.n

            # if eta <= 1e-10:  # can be smaller?
            #     update = True

            if -self.quasi_phi(a=1) <= f + np.sum(g * (self.x_ - self.x)) + np.sum((self.x_ - self.x) ** 2) / (2 * eta):
                update = True
            else:
                eta = factor[1] * eta
                # print("-")
                shrinking_times += 1
            if update:
                self.p = self.p_.copy()
                self.x = self.x_.copy()
                self.u_rho = self._raw_u_rho(self.x)
                if self.sparse and np.any(self.u_rho_min > self.u_rho):
                    warnings.warn("The utility lower bound is not appropriate.")
                f = -self.quasi_phi()
                g = -self.deriv_quasi_phi()

                if shrinking_times == 0:
                    eta = factor[2] * eta
                    # print("+")

                shrinking_times = 0
                update = False

                self.store_processing_record(k + 1, cost, data, store=store, processing=processing, record=record,
                                             freq=(print_ / 20, print_, 1))

            k += 1

        # Do not expose a rejected final line-search proposal as the tentative
        # state, and never alias tentative and accepted iterates.
        self.x_ = self.x.copy()
        self.p_ = self.p.copy()

        self.write('', 'end', processing=processing, record=record)
        if store:
            return data

    def solve_bcpr(self, num_iter, cyclic=False, processing=True, store=True, record=True, print_=50):
        self.initialize()
        setting = "------ BCPR ------\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        i = -1
        for k in range(1, num_iter + 1):
            if not cyclic:
                i = np.random.randint(self.n)
            else:
                i = (i + 1) % self.n

            x_i = np.divide(self.b[i], self.p, out=np.zeros_like(self.b[i]), where=self.p > 0)
            u_rho_i = np.dot(self.v[i], x_i ** self.rho)
            b_i = self.B[i] * self.v[i] * x_i ** self.rho / u_rho_i

            cost += self.m

            self.p = self.p + b_i - self.b[i]
            self.b[i] = b_i
            self.x = np.divide(
                self.b, self.p.reshape(1, -1), out=self.x.copy(),
                where=self.p.reshape(1, -1) > 0,
            )  # do not need in algorithm, only for evaluation

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))

        self.write('', 'end', processing=processing, record=record)
        if store:
            return data

    def solve_pr(self, num_iter, processing=True, store=True, record=True, print_=50, init=False):
        if init:
            sparse, self.sparse = self.sparse, False
            self.initialize()
            self.sparse = sparse
        else:
            self.initialize()
        setting = "------ PR ------\n" \
                  + f"- [the number of iteration] \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        for k in range(1, num_iter + 1):
            self.b = self.B.reshape(-1, 1) * ((self.v * self.x ** self.rho) / self.u_rho.reshape(-1, 1))
            self.p = np.sum(self.b, axis=0)
            self.x = np.divide(
                self.b, self.p.reshape(1, -1), out=self.x.copy(),
                where=self.p.reshape(1, -1) > 0,
            )
            self.u_rho = self._raw_u_rho(self.x)

            cost += self.n * self.m

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))
        self.write('', 'end', processing=processing, record=record)
        if store:
            return data
