 def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau: Union[float, ArrayType, Callable] = 8.0,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau = self.init_param(tau)

    # function
    self.integral = odeint(self.derivative, method=method)
    self._current = None

    self.reset_state(self.mode)

  def derivative(self, g, t):
    return -g / self.tau

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.g = self.init_variable(bm.zeros, batch_or_mode)


[docs]
  def update(self, x=None):
    self.g.value = self.integral(self.g.value, share['t'], share['dt'])
    if x is not None:
      self.add_current(x)
    return self.g.value



  def add_current(self, x):
    self.g.value += x

  def return_info(self):
    return self.g


def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau_decay: Union[float, ArrayType, Callable] = 10.0,
      tau_rise: Union[float, ArrayType, Callable] = 1.,
      A: Optional[Union[float, ArrayType, Callable]] = None,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_rise = self.init_param(tau_rise)
    self.tau_decay = self.init_param(tau_decay)
    A = _format_dual_exp_A(self, A)
    self.a = (self.tau_decay - self.tau_rise) / self.tau_rise / self.tau_decay * A

    # integrator
    self.integral = odeint(JointEq(self.dg, self.dh), method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.h = self.init_variable(bm.zeros, batch_or_mode)
    self.g = self.init_variable(bm.zeros, batch_or_mode)

  def dh(self, h, t):
    return -h / self.tau_rise

  def dg(self, g, t, h):
    return -g / self.tau_decay + h


[docs]
  def update(self, x):
    # x: the pre-synaptic spikes

    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g.value, self.h.value, share['t'], dt=share['dt'])
    self.h += self.a * x
    return self.g.value



  def return_info(self):
    return self.g




