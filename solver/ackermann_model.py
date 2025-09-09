# -*- coding: utf-8 -*-
import casadi as ca

class AckermannModel:
    """Simple bicycle model: state x=[x,y,theta,v,delta], control u=[a, delta_cmd]"""
    def __init__(self, wheelbase=0.26, delta_rate=3.0):
        self.nx = 5
        self.nu = 2
        self.L = float(wheelbase)
        self.delta_rate = float(delta_rate)

    def forward(self, x, u, dt):
        x_, y_, th, v, de = x[0], x[1], x[2], x[3], x[4]
        a, dcmd = u[0], u[1]
        # first-order steer dynamics
        dde = ca.fmin(ca.fmax(dcmd - de, -self.delta_rate), self.delta_rate)
        de_next = de + dde * dt
        # kinematics
        beta = ca.atan(0.5 * ca.tan(de_next))
        x_next = x_ + v * ca.cos(th + beta) * dt
        y_next = y_ + v * ca.sin(th + beta) * dt
        th_next = th + (v / self.L) * ca.tan(de_next) * dt
        v_next = v + a * dt
        return ca.vertcat(x_next, y_next, th_next, v_next, de_next)
