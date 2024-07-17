class STDP(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']
    
    @staticmethod
    def derivative(s, A_s, A_t, t, tau, tau_s, tau_t):
        dsdt = -s / tau
        dAsdt = - A_s / tau_s
        dAtdt = - A_t / tau_t
        return dsdt, dAsdt, dAtdt
    
    def __init__(self, pre, post, conn, delay=0., 
                delta_A_s=0.5, delta_A_t=0.5, w_min=0., w_max=20., 
                tau_s=10., tau_t=10., tau=10., **kwargs):
        # parameters
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.tau = tau
        self.delta_A_s = delta_A_s
        self.delta_A_t = delta_A_t
        self.w_min = w_min
        self.w_max = w_max
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.A_s = bp.ops.zeros(self.size)
        self.A_t = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size) * 1.
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')
        
        super(STDP, self).__init__(pre=pre, post=post, **kwargs)


    def update(self, _t):
        for i in range(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]

            self.s[i], A_s, A_t = self.integral(self.s[i], self.A_s[i], self.A_t[i],
                                                _t, self.tau, self.tau_s, self.tau_t)

            w = self.w[i]
            if self.pre.spike[pre_id] > 0:
                self.s[i] += w
                A_s += self.delta_A_s
                w -= A_t

            if self.post.spike[post_id] > 0:
                A_t += self.delta_A_t
                w += A_s

            self.A_s[i] = A_s
            self.A_t[i] = A_t
            if w > self.w_max:
                w = self.w_max
            if w < self.w_min:
                w = self.w_min
            self.w[i] = w

            # output
            self.I_syn.push(i, self.s[i])
            self.post.input[post_id] += self.I_syn.pull(i)
pre = bm.neurons.LIF(1, monitors=['spike'])
post = bm.neurons.LIF(1, monitors=['spike'])

# pre before post
duration = 60.
(I_pre, _) = bp.inputs.constant_current([(0, 5), (30, 15), 
                                         (0, 5), (30, 15), 
                                         (0, duration-40)])
(I_post, _) = bp.inputs.constant_current([(0, 7), (30, 15), 
                                          (0, 5), (30, 15), 
                                          (0, duration-7-35)])

syn = STDP(pre=pre, post=post, conn=bp.connect.All2All(), monitors=['s', 'A_s', 'A_t', 'w'])
net = bp.Network(pre, syn, post)
net.run(duration, inputs=[(pre, 'input', I_pre), (post, 'input', I_post)])

# plot
fig, gs = bp.visualize.get_figure(3, 1)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.w[:, 0], label='w')
plt.legend()

fig.add_subplot(gs[2, 0])
plt.plot(net.ts, 2*pre.mon.spike[:, 0], label='pre_spike')
plt.plot(net.ts, 2*post.mon.spike[:, 0], label='post_spike')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label='s')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()


class STP(bp.TwoEndConn):
    target_backend = 'general'

    @staticmethod
    def derivative(s, u, x, t, tau, tau_d, tau_f):
        dsdt = -s / tau
        dudt = - u / tau_f
        dxdt = (1 - x) / tau_d
        return dsdt, dudt, dxdt
    
    def __init__(self, pre, post, conn, delay=0., U=0.15, tau_f=1500., tau_d=200., tau=8.,  **kwargs):
        # parameters
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.tau = tau
        self.U = U
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.ones(self.size)
        self.u = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size)
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')
        
        super(STP, self).__init__(pre=pre, post=post, **kwargs)


    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], u, x = self.integral(self.s[i], self.u[i], self.x[i], _t, self.tau, self.tau_d, self.tau_f)

            if self.pre.spike[pre_id] > 0:
                u += self.U * (1 - self.u[i])
                self.s[i] += self.w[i] * u * self.x[i]
                x -= u * self.x[i]
            self.u[i] = u
            self.x[i] = x

            # output
            post_id = self.post_ids[i]
            self.I_syn.push(i, self.s[i])
            self.post.input[post_id] += self.I_syn.pull(i)
neu1 = bm.neurons.LIF(1, monitors=['V'])
neu2 = bm.neurons.LIF(1, monitors=['V'])

# STD
syn = STP(U=0.2, tau_d=150., tau_f=2., pre=neu1, post=neu2, 
          conn=bp.connect.All2All(), monitors=['s', 'u', 'x'])
net = bp.Network(neu1, syn, neu2)
net.run(100., inputs=(neu1, 'input', 28.))

# plot
fig, gs = bp.visualize.get_figure(2, 1, 3, 7)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.u[:, 0], label='u')
plt.plot(net.ts, syn.mon.x[:, 0], label='x')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label='s')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()
