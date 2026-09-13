import cvxpy as cp
from scipy.special import kl_div
from functions import *
import warnings


class Linear:
    def __init__(self, v, bgt=None, sparse=False):
        self.n, self.m = v.shape
        self.index_list = np.arange(1, self.n + 1)
        self.index_matrix = np.arange(1, self.n + 1).reshape(-1, 1) * np.ones(self.m)
        if bgt is None:
            self.bgt = np.ones(shape=self.n)
        else:
            self.bgt = bgt
        self.v = v
        self.log_v = np.where(v > 0, np.log(v), 0)

        self.sparse = sparse
        self.v_one_cols = np.sum(self.v != 0, axis=0) != 1

        self.x = self.x_ = (self.bgt / np.sum(self.bgt)).reshape(-1, 1) * np.ones((self.n, self.m))
        self.u_min = self.u = np.sum(self.v * self.x, axis=1)

        self.cons_2 = - self.bgt / (2 * self.u_min ** 2)
        self.cons_1 = 2 * self.bgt / self.u_min
        self.cons_0 = self.bgt * np.log(self.u_min) - 3 / 2 * self.bgt
        self.dual_cons = sum(self.bgt * np.log(self.bgt) - self.bgt)

        self.p = self.p_ = (np.sum(self.bgt) / self.m) * np.ones(self.m)
        self.beta = self.bgt / self.u
        self.b = self.b_ = (self.bgt / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        self.opt_x, self.opt_u, self.opt_p, self.opt_b = self.x, self.u, self.p, self.b
        self.file = None

    def initialize(self, alpha=0.06):
        if not self.sparse:
            self.x = self.x_ = (self.bgt / np.sum(self.bgt)).reshape(-1, 1) * np.ones((self.n, self.m))
            self.u = np.sum(self.v * self.x, axis=1)
            self.p = self.p_ = (np.sum(self.bgt) / self.m) * np.ones(self.m)
            self.beta = self.bgt / self.u
            self.b = self.b_ = (self.bgt / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        else:
            self.solve_pr(num_iter=1, record=False, processing=False, init=True)
            self.x_ = self.x
            self.u_min = alpha * self.u
            self.cons_2 = self.bgt / self.u_min ** 2
            self.cons_1 = 2 * self.bgt / self.u_min
            self.cons_0 = self.bgt * np.log(self.u_min) - 1.5 * self.bgt

    def phi(self):
        u = np.sum(self.v * self.x, axis=1)

        return np.sum(self.bgt * np.log(u))

    def dual_phi(self):
        self.u = np.sum(self.v * self.x, axis=1)
        self.beta = self.bgt / self.u
        p = np.amax(self.beta.reshape(-1, 1) * self.v, axis=0)

        return sum(p) - sum(self.bgt * np.log(self.beta)) + self.dual_cons

    def quasi_phi(self, a=None):
        if a is None:
            u = np.sum(self.v * self.x, axis=1)
            f = np.where(u >= self.u_min, self.bgt * np.log(u),
                         self.cons_0 + self.cons_1 * u + self.cons_2 * u ** 2)
        else:
            u = np.sum(self.v * self.x_, axis=1)
            f = np.where(u >= self.u_min, self.bgt * np.log(u),
                         self.cons_0 + self.cons_1 * u + self.cons_2 * u ** 2)

        return np.sum(f)

    def deriv_quasi_phi(self):
        u = np.sum(self.v * self.x, axis=1)
        g = np.where(u < self.u_min, self.cons_1 + 2 * self.cons_2 * u, self.bgt / u)

        return g.reshape(-1, 1) * self.v

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
        p = np.amax((self.bgt / u).reshape(-1, 1) * self.v, axis=0)
        p_gap = np.absolute(p - self.opt_p) / self.opt_p
        if mode == 'all':
            return p_gap
        elif mode == 'max':
            return max(p_gap)
        elif mode == 'avg':
            return np.average(p_gap)

    def shmyrev_obj(self):

        return sum(np.sum(self.log_v * self.b, axis=0)) - sum(self.p * np.log(self.p))

    def record(self):
        self.file = open(f'./records/record-linear-{self.n}-{self.m}', "w", encoding='utf-8')
        self.file.write(f"the number of buyers: {self.n} \n"
                        f"the number of goods: {self.m} \n")
        self.file.write("the valuation matrix: \n")
        for row in self.v:
            self.file.write(str(row) + "\n")
        self.file.write("the budget of buyers: \n" + str(self.bgt) + "\n\n\n")

    def write(self, content, pos, processing=True, record=True):
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

    def solve_opt_cvxpy(self, processing=True, record=True):
        x = cp.Variable((self.n, self.m), nonneg=True)
        w = np.ones(self.n)
        obj = cp.Maximize(sum([self.bgt[i] * cp.log(cp.matmul(self.v[i], x[i])) for i in range(self.n)]))
        constraints = [w @ x <= 1]
        prob = cp.Problem(obj, constraints)
        prob.solve('MOSEK')
        if processing:
            print("> solve primal problem with cvxpy (solver='MOSEK')")
        if prob.status == 'optimal':
            self.opt_x = self.x = x.value
            self.opt_u = np.sum(self.v * self.opt_x, axis=1)
            self.opt_p = np.amax((self.bgt / self.opt_u).reshape(-1, 1) * self.v, axis=0)
            self.opt_b = self.opt_x * self.opt_p
            if processing:
                print("> SOLVED!")
                print(f"> optimal dual gap: {self.dual_gap()}\n")
            if record:
                self.file.write(f"> optimal solution: \n{self.x}")

    def solve_bcdeg(self, num_iter, alpha=0.10, eta0=1, factor=(0.80, 1.02), step_size='fixed_step', cyclic=False,
                    processing=True, store=True, record=True, print_=50):
        self.initialize(alpha=alpha)
        setting = "------ BCDEG ------\n" \
                  + f"- [step size strategy]    \t{step_size}\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        j = 0
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
                lip_j = max(self.bgt * self.v[:, j] ** 2 / u_minus_one ** 2)
                eta[j] = 1 / lip_j

            update = False
            if self.sparse and not self.v_one_cols[j]:
                cost += self.n
            else:
                g = np.where(self.u < self.u_min, (self.cons_1 + 2 * self.cons_2 * self.u) * self.v[:, j],
                             self.bgt * self.v[:, j] / self.u)
                d0 = self.x[:, j] + eta[j] * g
                p_j = compute_for_price(d0, eta[j], self.index_list)
                x_j = np.maximum(d0 - eta[j] * p_j, 0)
                u_ = self.u + self.v[:, j] * (x_j - self.x[:, j])

                if self.sparse and sum(self.u_min > u_) > 0:
                    warnings.warn("The utility lower bound is not appropriate.")

                cost += self.n

                if step_size == 'line_search':
                    g_j = np.where(u_ < self.u_min, (self.cons_1 + 2 * self.cons_2 * u_) * self.v[:, j],
                                   self.bgt * self.v[:, j] / u_)
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
        if store:
            return data

    def solve_bcpr(self, num_iter, delta=0.01, eta0=1, factor=(0.80, 1.02), step_size='fixed_step', cyclic=False,
                   processing=True, store=True, record=True, print_=50):
        self.initialize()
        setting = "------ BCPR ------\n" \
                  + f"- [step size strategy]    \t{step_size}\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        i = 0
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
                v_p_i = (self.v[i] / self.p)[self.opt_b[i] >= 1e-200]
                v_p_i_nonzero = v_p_i[v_p_i > 0]
                beta_i = max(v_p_i_nonzero) / min(v_p_i_nonzero)
                theta_i = max(self.b[i] / self.p)
                if beta_i <= np.sqrt(2):
                    lip_i = (3 / (4 - beta_i)) * (theta_i + (1 - (0.5/beta_i)) / 3 * theta_i ** 2) / 5
                else:
                    lip_i = (3 / (4 - beta_i)) * (theta_i + (1 / 3) * theta_i ** 2) / 5
                eta[i] = (1 - delta) * min(max(1 / lip_i, 1), eta_bar)

            update = False
            eta_i = (1 - delta) * eta[i]
            x_i = self.b[i] / self.p ** eta_i
            u_i = np.dot(self.v[i] ** eta_i, x_i)
            b_i = self.bgt[i] * self.v[i] ** eta_i * x_i / u_i
            p_ = self.p + b_i - self.b[i]

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
                self.x = self.b / self.p  # do not need in algorithm, only for evaluation

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))

        self.write('', 'end', processing=processing, record=record)
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

            if self.sparse:
                d0 = d0[:, self.v_one_cols]
                self.p_[self.v_one_cols] = compute_for_price_all(d0, eta, self.index_matrix[:, self.v_one_cols])
                self.x_[:, self.v_one_cols] = np.maximum(d0 - eta * self.p_[self.v_one_cols], 0)
            else:
                self.p_ = compute_for_price_all(d0, eta, self.index_matrix)
                self.x_ = np.maximum(d0 - eta * self.p_, 0)

            cost += self.n * self.m

            if eta <= 1e-5:
                update = True

            if -self.quasi_phi(a=1) <= f + np.sum(g * (self.x_ - self.x)) + np.sum((self.x_ - self.x) ** 2) / (2 * eta):
                update = True
            else:
                eta = factor[1] * eta
                shrinking_times += 1

            if update:
                self.p = self.p_
                self.x = self.x_
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

            sum_ = np.sum(self.b * (self.v / self.p) ** eta, axis=1)
            self.b_ = self.bgt.reshape(-1, 1) * self.b * (self.v / self.p) ** eta / sum_.reshape(-1, 1)
            self.p_ = np.sum(self.b_, axis=0)

            cost += self.n * self.m

            if eta * sum(kl_div(self.p_, self.p)) <= np.sum(kl_div(self.b_, self.b)):
                update = True
            else:
                eta = max(factor[1] * eta, 1)
                shrinking_times += 1

            if update:
                self.b = self.b_
                self.p = self.p_
                self.x = self.b / self.p

                if shrinking_times == 0:
                    eta = factor[2] * eta
                shrinking_times = 0
                update = False

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))
        self.write('', 'end', processing=processing, record=record)
        if store:
            return data
