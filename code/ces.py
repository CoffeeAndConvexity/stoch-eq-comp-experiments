import cvxpy as cp
import numpy as np
from scipy.special import kl_div
from functions import *
import warnings


class CES:
    def __init__(self, v, rho, B=None, sparse=False):
        self.n, self.m = v.shape
        self.rho = rho
        self.index_list = np.arange(1, self.n + 1)
        self.index_matrix = np.arange(1, self.n + 1).reshape(-1, 1) * np.ones(self.m)
        if B is None:
            self.B = np.ones(shape=self.n)
        else:
            self.B = B
        self.v = v

        self.sparse = sparse
        self.v_one_cols = np.sum(self.v != 0, axis=0) != 1
        self.v_nonzero = (self.v != 0)

        self.x = self.x_ = (self.B / np.sum(self.B)).reshape(-1, 1) * np.ones((self.n, self.m))
        self.u_rho = np.sum(self.v * self.x ** self.rho, axis=1)
        # print(self.u_rho)
        self.u_rho_min = self.u_rho
        delta_1 = self.B / (self.m * sum(self.B))
        min_v = np.amin(self.v, axis=1)
        max_v = np.amax(self.v, axis=1)
        delta_2 = min_v / max_v
        self.x_min = (delta_1 ** (1 / (1 - self.rho)) * delta_2 ** (
                self.rho * (self.rho + 1) / (1 - self.rho))).reshape(-1, 1) \
                     * np.ones(self.m)

        self.cons_2_x = self.rho * (self.rho - 1) / 2 * self.v * self.x_min ** (self.rho - 2)
        self.cons_1_x = self.rho * (2 - self.rho) * self.v * self.x_min ** (self.rho - 1)
        self.cons_0_x = (1 - self.rho / 2) * (1 - self.rho) * self.v * self.x_min ** self.rho

        self.cons_2 = - self.B / (2 * self.rho * self.u_rho_min ** 2)
        self.cons_1 = 2 * self.B / (self.rho * self.u_rho_min)
        self.cons_0 = self.B / self.rho * np.log(self.u_rho_min) - 3 / 2 * self.B / self.rho

        self.b = (self.B / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        self.p = self.p_ = (np.sum(self.B) / self.m) * np.ones(self.m)
        self.beta = self.B / self.u_rho ** (1 / self.rho)
        self.opt_x, self.opt_u, self.opt_p, self.opt_b = self.x, self.u_rho ** (1 / self.rho), self.p, self.b
        self.file = None

    def initialize(self, alpha=0.06):
        if not self.sparse:
            self.x = self.x_ = (self.B / np.sum(self.B)).reshape(-1, 1) * np.ones((self.n, self.m))
            self.u_rho = np.sum(self.v * self.x ** self.rho, axis=1)
            self.p = self.p_ = (np.sum(self.B) / self.m) * np.ones(self.m)
            self.beta = self.B / self.u_rho ** (1 / self.rho)
            self.b = (self.B / self.m).reshape(-1, 1) * np.ones((self.n, self.m))
        else:
            self.solve_pr(num_iter=1, record=False, processing=False, init=True)
            self.x_ = self.x
            self.u_rho_min = alpha * self.u_rho
            self.cons_2 = - self.B / (2 * self.rho * self.u_rho_min ** 2)
            self.cons_1 = 2 * self.B / (self.rho * self.u_rho_min)
            self.cons_0 = self.B / self.rho * np.log(self.u_rho_min) - 3 / 2 * self.B / self.rho

    def phi(self):
        u_rho = np.sum(self.v * self.x ** self.rho, axis=1)

        return np.sum(self.B / self.rho * np.log(u_rho))

    def dual_phi(self):
        u = np.sum(self.v * self.x ** self.rho, axis=1) ** (1 / self.rho)
        beta = self.B / u
        p = np.amax((beta * u ** (1 - self.rho)).reshape(-1, 1) * self.v * self.x ** (self.rho - 1), axis=0)

        return sum(p) + sum(self.B * (-1 - np.log(beta) + np.log(self.B))) \
               + self.compute_dual_sub_opt(beta, p)

    def compute_dual_sub_opt(self, beta, p):
        x = cp.Variable((self.n, self.m), nonneg=True)
        term1 = sum([beta[i] * cp.pnorm(cp.matmul(self.v[i] ** (1 / self.rho), x[i]), self.rho) for i in range(self.n)])
        obj = cp.Maximize(term1 - sum(x @ p))
        constraints = []

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve('MOSEK')
            if prob.status == 'optimal':
                return prob.value
            else:
                print("compute_dual_sub_opt not optimal")
        except:
            return np.nan

    def quasi_phi(self, a=None):
        u_rho = self.quasi_u_rho(a=a)
        f = np.where(u_rho >= self.u_rho_min, self.B / self.rho * np.log(u_rho),
                     self.cons_0 + self.cons_1 * u_rho + self.cons_2 * u_rho ** 2)

        return np.sum(f)

    def deriv_quasi_phi(self):
        u_rho = self.quasi_u_rho()
        g = np.where(u_rho >= self.u_rho_min, self.B / u_rho, (self.cons_1 + 2 * self.cons_2 * u_rho) * self.rho)

        return g.reshape(-1, 1) / self.rho * self.deriv_quasi_u_rho()

    def quasi_u_rho(self, a=None):
        if a is None:
            inp_v_x_rho = np.where(self.x >= self.x_min, self.v * self.x ** self.rho,
                                   self.cons_0_x + self.cons_1_x * self.x + self.cons_2_x * self.x ** 2)
        else:
            inp_v_x_rho = np.where(self.x_ >= self.x_min, self.v * self.x_ ** self.rho,
                                   self.cons_0_x + self.cons_1_x * self.x_ + self.cons_2_x * self.x_ ** 2)

        return np.sum(inp_v_x_rho, axis=1)

    def deriv_quasi_u_rho(self):
        # print(self.x)
        # print(np.where(self.x >= self.x_min, self.rho * self.v * self.x ** (self.rho - 1),
        #                self.cons_1_x + 2 * self.cons_2_x * self.x))

        return np.where(self.x >= self.x_min, self.rho * self.v * self.x ** (self.rho - 1),
                        self.cons_1_x + 2 * self.cons_2_x * self.x)

    def dual_gap(self):
        obj_primal = self.phi()
        obj_dual = self.dual_phi()

        return obj_dual - obj_primal

    def utility_gap(self, mode='all'):
        u = np.sum(self.v * self.x ** self.rho, axis=1) ** (1 / self.rho)
        u_gap = np.absolute(u - self.opt_u) / self.opt_u
        if mode == 'all':
            return u_gap
        elif mode == 'max':
            return max(u_gap)
        elif mode == 'avg':
            return np.average(u_gap)

    def price_gap(self, mode='all'):
        u = np.sum(self.v * self.x ** self.rho, axis=1) ** (1 / self.rho)
        beta = self.B / u
        p = np.amax((beta * u ** (1 - self.rho)).reshape(-1, 1) * self.v * self.x ** (self.rho - 1), axis=0)
        p_gap = np.absolute(p - self.opt_p) / self.opt_p
        if mode == 'all':
            return p_gap
        elif mode == 'max':
            return max(p_gap)
        elif mode == 'avg':
            return np.average(p_gap)

    def record(self):
        self.file = open(f'./records/record-linear-{self.n}-{self.m}', "w", encoding='utf-8')
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
        obj = cp.Maximize(sum([self.B[i] / self.rho * cp.log(cp.matmul(self.v[i], x[i] ** self.rho))
                               for i in range(self.n)]))
        constraints = [w @ x <= 1]
        prob = cp.Problem(obj, constraints)
        prob.solve('MOSEK')
        if processing:
            print("> solve primal problem with cvxpy (solver='MOSEK')")
        if prob.status == 'optimal':
            self.opt_x = self.x = x.value
            self.opt_u = np.sum(self.v * self.opt_x ** self.rho, axis=1) ** (1 / self.rho)
            self.opt_p = np.amax((self.B * self.opt_u ** (- self.rho)).reshape(-1, 1) *
                                 self.v * self.opt_x ** (self.rho - 1), axis=0)
            if processing:
                print("> SOLVED!")
                print(f"> optimal primal value: {self.phi()}\n")
                print(f"> optimal dual gap: {self.dual_gap()}\n")
            if record:
                self.file.write(f"> optimal solution: \n{self.x}")

    def solve_bcdeg(self, num_iter, alpha=0.10, eta0=1, factor=(1, 0.80, 1.02), step_size='fixed_step',
                    cyclic=False, processing=True, store=True, record=True, print_=50):
        self.initialize(alpha=alpha)
        setting = "------ BCDEG ------\n" \
                  + f"- [step size strategy]    \t{step_size}\n" \
                  + f"- [the number of iteration]   \t{num_iter}"
        self.write(setting, 'begin', processing=processing, record=record)
        data, cost = create_data()
        store_data(data, cost, self.phi(), self.dual_phi(), self.utility_gap(), self.price_gap())

        j = 0
        lip_j = np.amax(self.B.reshape(-1, 1) * self.v * self.x_min ** (self.rho - 2) / self.u_rho_min.reshape(-1, 1),
                        axis=0)
        eta = eta0 * (1 / lip_j)

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
                u_rho_minus1 = self.u_rho - self.v[:, j] * self.x[:, j] ** self.rho
                lip_j = max(self.B * self.v[:, j] / u_rho_minus1) / self.x_min ** (2 - self.rho)
                eta[j] = 1 / lip_j

            update = False

            if self.sparse and not self.v_one_cols[j]:
                cost += self.n
            else:
                g = np.where(self.u_rho >= self.u_rho_min, self.B / self.u_rho,
                             (self.cons_1 + 2 * self.cons_2 * self.u_rho) * self.rho)
                g_part2 = np.where(self.x[:, j] >= self.x_min[:, j], self.v[:, j] * self.x[:, j] ** (self.rho - 1),
                                   (self.cons_1_x[:, j] + 2 * self.cons_2_x[:, j] * self.x[:, j]) / self.rho)
                g = g * g_part2

                d0 = self.x[:, j] + eta[j] * g
                p_j = compute_for_price(d0, eta[j], self.index_list)
                x_j = np.maximum(d0 - eta[j] * p_j, 0)
                u_rho_ = self.u_rho + self.v[:, j] * (x_j ** self.rho - self.x[:, j] ** self.rho)

                if self.sparse and sum(self.u_rho_min > u_rho_) > 0:
                    warnings.warn("The utility lower bound is not appropriate.")

                cost += self.n

                if step_size == 'line_search':
                    g_j = np.where(u_rho_ >= self.u_rho_min, self.B / u_rho_,
                                   (self.cons_1 + 2 * self.cons_2 * u_rho_) * self.rho)
                    g_j_part2 = np.where(x_j >= self.x_min[:, j], self.v[:, j] * x_j ** (self.rho - 1),
                                         (self.cons_1_x[:, j] + 2 * self.cons_2_x[:, j] * x_j) / self.rho)
                    g_j = g_j * g_j_part2
                    if sum((g_j - g) ** 2) > (1 / eta[j]) ** 2 * sum((x_j - self.x[:, j]) ** 2):
                        # print(j)
                        # print(sum((g_j - g) ** 2))
                        # print(g_part2, g_j_part2)
                        # print((1 / eta[j]) ** 2 * sum((x_j - self.x[:, j]) ** 2))
                        j_change = False
                        eta[j] = eta[j] * factor[1]
                    else:
                        # print("ke", j)
                        # print(sum((g_j - g) ** 2))
                        # print(g_part2, g_j_part2)
                        # print((1 / eta[j]) ** 2 * sum((x_j - self.x[:, j]) ** 2))
                        update = True
                        j_change = True
                        eta[j] = eta[j] * factor[2]

                if update or (step_size == 'fixed_step') or (step_size == 'adaptive'):
                    self.x[:, j] = x_j
                    self.u_rho = u_rho_
                    self.p[j] = p_j

                    # print(self.phi())
                    # print(self.quasi_phi())
                    # print()

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
            if self.sparse:
                d0 = d0[:, self.v_one_cols]
                self.p_[self.v_one_cols] = compute_for_price_all(d0, eta, self.index_matrix[:, self.v_one_cols])
                self.x_[:, self.v_one_cols] = np.maximum(d0 - eta * self.p_[self.v_one_cols], 0)
            else:
                self.p_ = compute_for_price_all(d0, eta, self.index_matrix)
                self.x_ = np.maximum(d0 - eta * self.p_, 0)

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
                self.p = self.p_
                self.x = self.x_
                self.u_rho = np.sum(self.v * self.x ** self.rho, axis=1)
                if self.sparse and sum(self.u_rho_min > self.u_rho) > 0:
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

        i = 0
        for k in range(1, num_iter + 1):
            if not cyclic:
                i = np.random.randint(self.n)
            else:
                i = (i + 1) % self.n

            x_i = self.b[i] / self.p
            u_rho_i = np.dot(self.v[i], x_i ** self.rho)
            b_i = self.B[i] * self.v[i] * x_i ** self.rho / u_rho_i

            cost += self.m

            self.p = self.p + b_i - self.b[i]
            self.b[i] = b_i
            self.x = self.b / self.p  # do not need in algorithm, only for evaluation

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
            self.x = self.b / self.p
            self.u_rho = np.sum(self.v * self.x ** self.rho, axis=1)

            cost += self.n * self.m

            self.store_processing_record(k, cost, data, store=store, processing=processing, record=record,
                                         freq=(print_ / 20, print_, 1))
        self.write('', 'end', processing=processing, record=record)
        if store:
            return data
