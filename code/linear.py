import cvxpy as cp
from scipy.special import kl_div, xlogy
from functions import *
import os
import warnings


class Linear:
    def __init__(self, v, bgt=None, sparse=False):
        self.v = np.array(v, dtype=float, copy=True)
        if self.v.ndim != 2 or 0 in self.v.shape:
            raise ValueError("v must be a nonempty two-dimensional valuation matrix.")
        if not np.all(np.isfinite(self.v)) or np.any(self.v < 0):
            raise ValueError("Valuations must be finite and nonnegative.")

        self.n, self.m = self.v.shape
        self.index_list = np.arange(1, self.n + 1)
        self.index_matrix = np.arange(1, self.n + 1).reshape(-1, 1) * np.ones(self.m)
        if bgt is None:
            self.bgt = np.ones(shape=self.n)
        else:
            self.bgt = np.array(bgt, dtype=float, copy=True)
        if self.bgt.shape != (self.n,):
            raise ValueError(f"bgt must have shape ({self.n},).")
        if not np.all(np.isfinite(self.bgt)) or np.any(self.bgt <= 0):
            raise ValueError("Budgets must be finite and strictly positive.")

        blank_buyers = np.flatnonzero(np.all(self.v == 0, axis=1))
        if blank_buyers.size:
            raise ValueError(
                "Every buyer must value at least one good; zero-valuation buyer "
                f"rows: {blank_buyers.tolist()}."
            )
        blank_goods = np.flatnonzero(np.all(self.v == 0, axis=0))
        if blank_goods.size:
            raise ValueError(
                "Every good must be valued by at least one buyer; zero-valuation "
                f"good columns: {blank_goods.tolist()}."
            )

        self.log_v = np.zeros_like(self.v)
        np.log(self.v, out=self.log_v, where=self.v > 0)

        self.sparse = bool(sparse)
        self.v_one_cols = np.sum(self.v != 0, axis=0) != 1

        self.x = (self.bgt / np.sum(self.bgt)).reshape(-1, 1) * np.ones((self.n, self.m))
        self.x_ = self.x.copy()
        self.u = np.sum(self.v * self.x, axis=1)
        self.u_min = self.u.copy()

        self._set_smoothing_coefficients()
        self.dual_cons = sum(self.bgt * np.log(self.bgt) - self.bgt)

        self.p = (np.sum(self.bgt) / self.m) * np.ones(self.m)
        self.p_ = self.p.copy()
        self.beta = self.bgt / self.u
        self.b = (self.bgt / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        self.b_ = self.b.copy()
        self.opt_x = self.x.copy()
        self.opt_u = self.u.copy()
        self.opt_p = self.p.copy()
        self.opt_b = self.b.copy()
        self.file = None

    def _set_smoothing_coefficients(self):
        """Set the C2 quadratic continuation of B_i log(u_i)."""
        self.cons_2 = -self.bgt / (2 * self.u_min ** 2)
        self.cons_1 = 2 * self.bgt / self.u_min
        self.cons_0 = self.bgt * np.log(self.u_min) - 1.5 * self.bgt

    @staticmethod
    def _validate_alpha(alpha):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must satisfy 0 < alpha <= 1.")

    def _allocation_from_bids(self):
        if np.any(~np.isfinite(self.p)) or np.any(self.p <= 0):
            raise FloatingPointError(
                "Bid updates produced a nonpositive or non-finite price. "
                "Check the valuation matrix and algorithm parameters."
            )
        return self.b / self.p

    def _adaptive_bcpr_step(self, i, eta_bar):
        """Return the Algorithm 7 adaptive step, or the safe unit fallback."""
        ratios = np.divide(
            self.v[i], self.p, out=np.zeros(self.m), where=self.p > 0
        )
        support = (self.opt_b[i] >= 1e-200) & np.isfinite(ratios) & (ratios > 0)
        if not np.any(support):
            # An inaccurate reference solve (or a caller-supplied reference)
            # can have an empty numerical support. Use all valued goods.
            support = np.isfinite(ratios) & (ratios > 0)
        if not np.any(support):
            return 1.0

        supported_ratios = ratios[support]
        ratio_i = np.max(supported_ratios) / np.min(supported_ratios)
        # Appendix C, (75): beta_i depends on the largest allowed exponent.
        # If this leaves Lemma 8's beta <= sqrt(2) regime, alpha=1 remains the
        # globally safe relative-smoothness step.
        with np.errstate(over='ignore', invalid='ignore'):
            beta_i = ratio_i ** eta_bar
        spending_shares = np.divide(
            self.b[i], self.p, out=np.zeros(self.m), where=self.p > 0
        )
        finite_shares = spending_shares[np.isfinite(spending_shares)]
        if finite_shares.size == 0:
            return 1.0
        theta_i = np.max(finite_shares)

        if not np.isfinite(beta_i) or beta_i > np.sqrt(2) or theta_i <= 0:
            return 1.0
        # Appendix C, (78).  (1 - 1/(2 beta))/3 is algebraically the same as
        # (2 beta - 1)/(6 beta).
        lip_i = (3 / (4 - beta_i)) * (
            theta_i + (1 - (0.5 / beta_i)) / 3 * theta_i ** 2
        )
        if not np.isfinite(lip_i) or lip_i <= 0:
            return 1.0
        return np.clip(1 / lip_i, 1, eta_bar)

    def _quasi_values(self, u):
        values = self.cons_0 + self.cons_1 * u + self.cons_2 * u ** 2
        regular = u >= self.u_min
        values[regular] = self.bgt[regular] * np.log(u[regular])
        return values

    def _quasi_marginal(self, u):
        marginal = self.cons_1 + 2 * self.cons_2 * u
        regular = u >= self.u_min
        marginal[regular] = self.bgt[regular] / u[regular]
        return marginal

    def initialize(self, alpha=0.06):
        self._validate_alpha(alpha)
        if not self.sparse:
            # The safe dense-market lower bound is the utility of the
            # proportional allocation; alpha is only used by the sparse-data
            # initialization below.
            self.x = (self.bgt / np.sum(self.bgt)).reshape(-1, 1) * np.ones((self.n, self.m))
            self.x_ = self.x.copy()
            self.u = np.sum(self.v * self.x, axis=1)
            self.u_min = self.u.copy()
            self._set_smoothing_coefficients()
            self.p = (np.sum(self.bgt) / self.m) * np.ones(self.m)
            self.p_ = self.p.copy()
            self.beta = self.bgt / self.u
            self.b = (self.bgt / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
            self.b_ = self.b.copy()
        else:
            self.solve_pr(num_iter=1, record=False, processing=False, init=True)
            self.x_ = self.x.copy()
            self.p_ = self.p.copy()
            self.b_ = self.b.copy()
            self.u_min = alpha * self.u
            self._set_smoothing_coefficients()

    def phi(self):
        u = np.sum(self.v * self.x, axis=1)
        if np.any(u <= 0):
            return -np.inf
        return np.sum(self.bgt * np.log(u))

    def dual_phi(self):
        self.u = np.sum(self.v * self.x, axis=1)
        if np.any(self.u <= 0):
            return np.inf
        self.beta = self.bgt / self.u
        p = np.amax(self.beta.reshape(-1, 1) * self.v, axis=0)

        return sum(p) - sum(self.bgt * np.log(self.beta)) + self.dual_cons

    def quasi_phi(self, a=None):
        if a is None:
            u = np.sum(self.v * self.x, axis=1)
        else:
            u = np.sum(self.v * self.x_, axis=1)
        return np.sum(self._quasi_values(u))

    def deriv_quasi_phi(self):
        u = np.sum(self.v * self.x, axis=1)
        return self._quasi_marginal(u).reshape(-1, 1) * self.v

    def dual_gap(self):
        obj_primal = self.phi()
        obj_dual = self.dual_phi()

        return obj_dual - obj_primal

    def utility_gap(self, mode='all'):
        u = np.sum(self.v * self.x, axis=1)
        u_gap = np.absolute(u - self.opt_u) / self.opt_u
        if mode == 'all':
            return u_gap
        elif mode == 'max':
            return max(u_gap)
        elif mode == 'avg':
            return np.average(u_gap)

    def price_gap(self, mode='all'):
        u = np.sum(self.v * self.x, axis=1)
        if np.any(u <= 0):
            p_gap = np.full(self.m, np.inf)
            if mode == 'all':
                return p_gap
            return np.inf
        p = np.amax((self.bgt / u).reshape(-1, 1) * self.v, axis=0)
        p_gap = np.absolute(p - self.opt_p) / self.opt_p
        if mode == 'all':
            return p_gap
        elif mode == 'max':
            return max(p_gap)
        elif mode == 'avg':
            return np.average(p_gap)

    def shmyrev_obj(self):
        return sum(np.sum(self.log_v * self.b, axis=0)) - sum(xlogy(self.p, self.p))

    def record(self):
        os.makedirs('./records', exist_ok=True)
        if self.file is not None and not self.file.closed:
            self.file.close()
        self.file = open(f'./records/record-linear-{self.n}-{self.m}', "w", encoding='utf-8')
        self.file.write(f"the number of buyers: {self.n} \n"
                        f"the number of goods: {self.m} \n")
        self.file.write("the valuation matrix: \n")
        for row in self.v:
            self.file.write(str(row) + "\n")
        self.file.write("the budget of buyers: \n" + str(self.bgt) + "\n\n\n")

    def _ensure_record_file(self):
        if self.file is None or self.file.closed:
            self.record()

    def write(self, content, pos, processing=True, record=True):
        if record:
            self._ensure_record_file()
        if pos == 'begin':
            if processing:
                print(content)
            if record:
                self.file.write(content + "\n")
        elif pos == 'end':
            if processing:
                print("------ Done! ------\n")
            if record:
                self.file.write("\n***{ Solution }***\n" + "allocation: \n")
                for row in self.x:
                    self.file.write(str(np.round(row, 4)) + "\n")
                self.file.write(f"primal objective: {self.phi()}")
                self.file.write("\n\n\n")

    def close_file(self):
        if self.file is not None and not self.file.closed:
            self.file.close()

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
        w = np.ones(self.n)
        obj = cp.Maximize(sum([self.bgt[i] * cp.log(cp.matmul(self.v[i], x[i])) for i in range(self.n)]))
        constraints = [w @ x <= 1]
        prob = cp.Problem(obj, constraints)

        if solver is None:
            installed = set(cp.installed_solvers())
            solvers = [name for name in ('MOSEK', 'CLARABEL', 'ECOS', 'SCS') if name in installed]
            if not solvers:
                solvers = [None]
        else:
            solvers = [solver]

        attempts = []
        solved_with = None
        last_error = None
        for candidate in solvers:
            try:
                if candidate is None:
                    prob.solve()
                else:
                    prob.solve(solver=candidate)
            except Exception as error:
                last_error = error
                attempts.append(f"{candidate or 'CVXPY default'}: {error}")
                continue
            attempts.append(f"{candidate or 'CVXPY default'}: {prob.status}")
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solved_with = candidate or prob.solver_stats.solver_name
                break

        if solved_with is None or x.value is None:
            message = "Unable to compute a reference equilibrium (" + "; ".join(attempts) + ")."
            if last_error is not None:
                raise RuntimeError(message) from last_error
            raise RuntimeError(message)

        if processing:
            print(f"> solve primal problem with cvxpy (solver='{solved_with}', status='{prob.status}')")

        opt_x = np.asarray(x.value, dtype=float)
        opt_u = np.sum(self.v * opt_x, axis=1)
        if not np.all(np.isfinite(opt_x)) or np.any(opt_u <= 0):
            raise RuntimeError("The reference solver returned a non-finite or zero-utility solution.")
        opt_p = np.amax((self.bgt / opt_u).reshape(-1, 1) * self.v, axis=0)
        if not np.all(np.isfinite(opt_p)) or np.any(opt_p <= 0):
            raise RuntimeError("The reference solver returned nonpositive or non-finite prices.")

        self.opt_x = opt_x.copy()
        self.opt_u = opt_u.copy()
        self.opt_p = opt_p.copy()
        self.opt_b = self.opt_x * self.opt_p
        self.x = self.opt_x.copy()
        self.u = self.opt_u.copy()
        self.beta = self.bgt / self.u
        self.p = self.opt_p.copy()
        self.p_ = self.p.copy()
        self.b = self.opt_b.copy()
        self.b_ = self.b.copy()
        if processing:
            print("> SOLVED!")
            print(f"> optimal dual gap: {self.dual_gap()}\n")
        if record:
            self._ensure_record_file()
            self.file.write(f"> optimal solution: \n{self.x}")

    def solve_bcdeg(self, num_iter, alpha=0.10, eta0=1, factor=(0.80, 1.02), step_size='fixed_step', cyclic=False,
                    processing=True, store=True, record=True, print_=50):
        if step_size not in ('fixed_step', 'line_search', 'adaptive'):
            raise ValueError("step_size must be 'fixed_step', 'line_search', or 'adaptive'")
        self.initialize(alpha=alpha)
        setting = "------ BCDEG ------\n" \
                  + f"- [step size strategy]    \t{step_size}\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        # Start at -1 so a cyclic run visits block 0 first.
        j = -1
        lip = np.amax(self.bgt.reshape(-1, 1) * self.v ** 2 / self.u_min.reshape(-1, 1) ** 2, axis=0)
        eta = eta0 * (1 / lip)

        j_change = True
        for k in range(1, num_iter + 1):
            if j_change:
                if not cyclic:
                    j = np.random.randint(self.m)
                else:
                    j = (j + 1) % self.m

            if step_size == 'adaptive':
                u_minus_one = self.u - self.v[:, j] * self.x[:, j]
                # Quadratic extrapolation caps curvature when removing this
                # block would otherwise leave a buyer with zero utility.
                curvature_floor = np.maximum(u_minus_one, self.u_min)
                lip_j = max(self.bgt * self.v[:, j] ** 2 / curvature_floor ** 2)
                eta[j] = 1 / lip_j

            update = False
            if self.sparse and not self.v_one_cols[j]:
                cost += self.n
            else:
                g = self._quasi_marginal(self.u) * self.v[:, j]
                d0 = self.x[:, j] + eta[j] * g
                p_j = compute_for_price(d0, eta[j], self.index_list)
                x_j = np.maximum(d0 - eta[j] * p_j, 0)
                u_ = self.u + self.v[:, j] * (x_j - self.x[:, j])

                if self.sparse and sum(self.u_min > u_) > 0:
                    warnings.warn("The utility lower bound is not appropriate.")

                cost += self.n

                if step_size == 'line_search':
                    g_j = self._quasi_marginal(u_) * self.v[:, j]
                    if sum((g_j - g) ** 2) > (1 / eta[j]) ** 2 * sum((x_j - self.x[:, j]) ** 2):
                        eta[j] = eta[j] * factor[0]
                        j_change = False
                    else:
                        update = True
                        j_change = True
                        eta[j] = eta[j] * factor[1]

                if update or (step_size == 'fixed_step') or (step_size == 'adaptive'):
                    self.x[:, j] = x_j
                    self.u = u_
                    self.p[j] = p_j

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))

        self.write('', 'end', processing=processing, record=record)
        # Do not expose a rejected final line-search proposal as tentative
        # state, and never alias tentative and accepted iterates.
        self.x_ = self.x.copy()
        self.p_ = self.p.copy()
        if store:
            return data

    def solve_bcpr(self, num_iter, delta=0.01, eta0=1, factor=(0.80, 1.02), step_size='fixed_step', cyclic=False,
                   processing=True, store=True, record=True, print_=50):
        if step_size not in ('fixed_step', 'line_search', 'adaptive'):
            raise ValueError("step_size must be 'fixed_step', 'line_search', or 'adaptive'")
        self.initialize()
        setting = "------ BCPR ------\n" \
                  + f"- [step size strategy]    \t{step_size}\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        # Start at -1 so a cyclic run visits buyer 0 first.
        i = -1
        eta = eta0 * np.ones(self.n)
        eta_bar = 5

        i_change = True
        for k in range(1, num_iter + 1):
            if i_change:
                if not cyclic:
                    i = np.random.randint(self.n)
                else:
                    i = (i + 1) % self.n

            if step_size == 'adaptive':
                eta[i] = self._adaptive_bcpr_step(i, eta_bar)

            update = False
            # Algorithm 7 applies its clipped adaptive step directly.  The
            # delta margin belongs to the fixed/line-search variants only.
            eta_i = eta[i] if step_size == 'adaptive' else (1 - delta) * eta[i]
            x_i = self.b[i] / self.p ** eta_i
            u_i = np.dot(self.v[i] ** eta_i, x_i)
            if not np.isfinite(u_i) or u_i <= 0:
                raise FloatingPointError(
                    f"BCPR produced a nonpositive or non-finite normalization for buyer {i}."
                )
            b_i = self.bgt[i] * self.v[i] ** eta_i * x_i / u_i
            p_ = self.p + b_i - self.b[i]
            if np.any(~np.isfinite(p_)) or np.any(p_ <= 0):
                raise FloatingPointError("BCPR produced a nonpositive or non-finite price.")

            cost += self.m

            if step_size == 'line_search':
                kl_div_b_i = sum(kl_div(b_i, self.b[i]))
                if eta[i] * sum(kl_div(p_, self.p)) > kl_div_b_i:
                    eta[i] = np.maximum(eta[i] * factor[0], 1)
                    i_change = False
                    if eta[i] == 1.0:
                        update = True
                        i_change = True
                else:
                    update = True
                    i_change = True
                    eta[i] = np.minimum(eta[i] * factor[1], 10)

            if update or (step_size == 'fixed_step') or (step_size == 'adaptive'):
                self.b[i] = b_i
                self.p = p_
                self.x = self._allocation_from_bids()  # only needed for evaluation
                self.u = np.sum(self.v * self.x, axis=1)

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))

        self.write('', 'end', processing=processing, record=record)
        self.x_ = self.x.copy()
        self.p_ = self.p.copy()
        if store:
            return data

    def solve_pgls(self, num_iter, alpha=0.10, factor=(100, 0.80, 1.02), max_ls=10, processing=True, store=True,
                   record=True, print_=50):
        self.initialize(alpha=alpha)
        L = np.max(self.bgt / self.u_min ** 2)
        Lf_eg = L * np.max(np.linalg.norm(self.v, axis=1)) ** 2  # default: 2-norm
        eta = factor[0] / Lf_eg

        setting = "------ PGLS ------\n" \
                  + f"- [the number of iteration]                             \t{num_iter}\n" \
                  + f"- [the parameter in minimal utility]                    \t{alpha}\n" \
                  + f"- [the initial stepsize]                                \t{eta} ({factor[0]} / L)\n"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        f = -self.quasi_phi()
        g = -self.deriv_quasi_phi()
        update = False
        shrinking_times = 0

        for k in range(1, num_iter + 1):
            if shrinking_times > max_ls:
                update = True
            d0 = self.x - eta * g

            # A line-search trial must not mutate the accepted iterate. This
            # matters in sparse mode, where only a subset of columns changes.
            candidate_x = self.x.copy()
            candidate_p = self.p.copy()

            if self.sparse:
                active = self.v_one_cols
                if np.any(active):
                    d0_active = d0[:, active]
                    candidate_p[active] = compute_for_price_all(
                        d0_active, eta, self.index_matrix[:, active]
                    )
                    candidate_x[:, active] = np.maximum(
                        d0_active - eta * candidate_p[active], 0
                    )
            else:
                candidate_p = compute_for_price_all(d0, eta, self.index_matrix)
                candidate_x = np.maximum(d0 - eta * candidate_p, 0)

            self.p_ = candidate_p
            self.x_ = candidate_x

            cost += self.n * self.m

            if eta <= 1e-5:
                update = True

            if -self.quasi_phi(a=1) <= f + np.sum(g * (self.x_ - self.x)) + np.sum((self.x_ - self.x) ** 2) / (2 * eta):
                update = True
            else:
                eta = factor[1] * eta
                shrinking_times += 1

            if update:
                self.p = candidate_p.copy()
                self.x = candidate_x.copy()
                self.u = np.sum(self.v * self.x, axis=1)
                if self.sparse and sum(self.u_min > self.u) > 0:
                    warnings.warn("The utility lower bound is not appropriate.")
                f = -self.quasi_phi()
                g = -self.deriv_quasi_phi()

                if shrinking_times == 0:
                    eta = factor[2] * eta
                shrinking_times = 0
                update = False

                self.store_processing_record(k + 1, cost, data, store=store, processing=processing, record=record,
                                             freq=(print_ / 20, print_, 1))
        self.write('', 'end', processing=processing, record=record)
        self.x_ = self.x.copy()
        self.p_ = self.p.copy()
        if store:
            return data

    def solve_pr(self, num_iter, factor=(1, 0.80, 1.02), max_ls=10, processing=True, store=True, record=True,
                 print_=50, init=False):
        if init:
            sparse, self.sparse = self.sparse, False
            self.initialize()
            self.sparse = sparse
        else:
            self.initialize()

        setting = "------ PR ------\n" \
                  + f"- [the number of iteration] \t{num_iter}" \
                  + f"- [factor] \t{factor}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        eta = factor[0]
        update = False
        shrinking_times = 0

        for k in range(1, num_iter + 1):
            if shrinking_times > max_ls:
                update = True

            if np.any(~np.isfinite(self.p)) or np.any(self.p <= 0):
                raise FloatingPointError("PR encountered a nonpositive or non-finite price.")
            response = self.b * (self.v / self.p) ** eta
            sum_ = np.sum(response, axis=1)
            if np.any(~np.isfinite(sum_)) or np.any(sum_ <= 0):
                raise FloatingPointError("PR produced a nonpositive or non-finite normalization.")
            self.b_ = self.bgt.reshape(-1, 1) * response / sum_.reshape(-1, 1)
            self.p_ = np.sum(self.b_, axis=0)
            if np.any(~np.isfinite(self.p_)) or np.any(self.p_ <= 0):
                raise FloatingPointError("PR produced a nonpositive or non-finite price.")

            cost += self.n * self.m

            if eta * sum(kl_div(self.p_, self.p)) <= np.sum(kl_div(self.b_, self.b)):
                update = True
            else:
                eta = max(factor[1] * eta, 1)
                shrinking_times += 1

            if update:
                self.b = self.b_
                self.p = self.p_
                self.x = self._allocation_from_bids()
                self.u = np.sum(self.v * self.x, axis=1)

                if shrinking_times == 0:
                    eta = factor[2] * eta
                shrinking_times = 0
                update = False

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))
        self.write('', 'end', processing=processing, record=record)
        if store:
            return data
