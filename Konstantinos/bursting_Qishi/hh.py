class Oja(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    @staticmethod
    def derivative(w, t, gamma, r_pre, r_post):
        dwdt = gamma * (r_post * r_pre - r_post * r_post * w)
        return dwdt

    def __init__(self, pre, post, conn, delay=0.,
                 gamma=0.005, w_max=1., w_min=0.,
                 **kwargs):
        # params
        self.gamma = gamma
        self.w_max = w_max
        self.w_min = w_min
        # no delay in firing rate models

        # conns
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)
        
        # data
        self.w = bp.ops.ones(self.size) * 0.05

        self.integral = bp.odeint(f=self.derivative)
        super(Oja, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        post_r = bp.ops.zeros(self.post.size[0])
        for i in range(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            add = self.w[i] * self.pre.r[pre_id]
            post_r[post_id] += add
            self.w[i] = self.integral(
                self.w[i], _t, self.gamma,
                self.pre.r[pre_id], self.post.r[post_id])
        self.post.r = post_r
	
neu_pre_num = 2
neu_post_num = 2
dt = 0.02
bp.backend.set('numpy', dt=dt)

# build network
neu_pre = neu(neu_pre_num, monitors=['r'])
neu_post = neu(neu_post_num, monitors=['r'])

syn = Oja(pre=neu_pre, post=neu_post,
          conn=bp.connect.All2All(), monitors=['w'])

net = bp.Network(neu_pre, syn, neu_post)

# create input
current_mat_in = []
current_mat_out = []
current1, _ = bp.inputs.constant_current(
    [(2., 20.), (0., 20.)] * 3 + [(0., 20.), (0., 20.)] * 2)
current2, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
current3, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
current_mat_in = np.vstack((current1, current2))
current_mat_out = current3
current_mat_out = np.vstack((current_mat_out, current3))

# simulate network
net.run(duration=200.,
        inputs=[(neu_pre, 'r', current_mat_in.T, '='),
                (neu_post, 'r', current_mat_out.T)])

