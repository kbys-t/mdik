# coding:utf-8

import copy
import numpy as np

######################################################
class GaitGenerator:
    def __init__(self, grav, dt, com, leg_pos,
        offset=np.array([0.0, 0.0, 1.0]),
        swing=0.1,
        adjust=0.0,
        th_contact=10, th_time=0.0,
        T_ini=1.0,
        T_end=1.0,
        ):
        self.grav = grav
        self.dt = dt
        self.com_ref = com.copy()
        self.com_offset = offset.copy()
        self.leg_pos = copy.deepcopy(leg_pos)
        self.leg_dis = {key1: val[1] - self.leg_pos[[key2 for key2 in self.leg_pos.keys() if key2 != key1][0]][1] for key1, val in self.leg_pos.items()}
        self.sw_height = swing
        self.frq_des = np.sqrt(self.grav / self.com_offset[-1])
        #
        self.adjust_gain = adjust * self.dt
        self.th_contact = th_contact
        self.th_time = th_time
        self.T_ini = T_ini
        self.T_end = T_end
        #
        self.is_start = False
        self.t_elapsed = 0.0
        self.t_phase = 0.0

######################################################
    def swing(self, x_ini, x_end, T_, t_):
        # minimum jerk trajectory
        # time management (for z, time is also converted into array)
        T_array = np.zeros_like(x_ini) + T_
        T_array[-1] *= 0.5
        t_array = np.zeros_like(x_ini) + t_
        # position management (for z, apex is set)
        xs = x_ini
        xe = x_end
        h_ = xs[-1] + self.sw_height
        # for z
        if t_ <= 0.5 * T_:
            xe[-1] = h_
        else:
            xs[-1] = h_
            t_array[-1] -= 0.5 * T_
        # limit time
        r_ = np.clip(t_array / T_array, 0.0, 1.0)
        #
        return xs + (xe - xs) * (10.0 * np.power(r_, 3) - 15.0 * np.power(r_, 4) + 6.0 * np.power(r_, 5))

    def adjust(self, x_, x_tar):
        return x_ - self.adjust_gain * (x_ - x_tar)

######################################################
    def set(self, ui, ue, un, Ts, Td, Tn):
        # set dsp conds
        self.frq_dsp = 0.5 * np.pi / Td
        self._frq_ratio = self.frq_des / (self.frq_des**2 + self.frq_dsp**2)
        # set terminal dcm for dsp
        be = (un - ue) * np.exp(- self.frq_des * Tn)
        dcm_dsp = ue + be
        # set terminal dcm for ssp (initial dcm for dsp)
        du = ue - ui
        bi = be * np.exp(- self.frq_des * Td) + du * (np.exp(- self.frq_des * Td) * (1.0 - self._frq_ratio * self.frq_des) - (1.0 - self._frq_ratio * self.frq_dsp))
        dcm_ssp = ue + bi
        # set coefficient for desired dcm trajectory
        self._coeff_ssp = (dcm_ssp - ui) * np.exp(- self.frq_des * Ts)
        self._coeff_dsp = (dcm_dsp - ue + du * (1.0 - self._frq_ratio * self.frq_des)) * np.exp(- self.frq_des * Td)
        #
        self._ui = ui + self.com_offset

    def trj_dcm(self, t_):
        if self.gait == "ssp":
            # desired vrp is on the stance leg
            return self._coeff_ssp * np.exp(self.frq_des * t_) + self._ui
        elif self.gait == "dsp":
            # desired vrp moves from the past to new stance legs with sine curve
            ue = self.leg_pos[self.leg_stance].copy() + self.com_offset
            du = ue - self._ui
            sct = self.frq_des * np.sin(self.frq_dsp * t_) + self.frq_dsp * np.cos(self.frq_dsp * t_)
            #
            return self._coeff_dsp * np.exp(self.frq_des * t_) + ue - du * (1.0 - self._frq_ratio * sct)

    def trj_com(self, com, dcm_ref):
        return com - self.frq_des * (com - dcm_ref) * self.dt

######################################################
    def check(self, contact, T_, t_):
        if self.gait == "ssp":
            if self.cnt_footstep == 0:
                return t_ >= T_
            elif t_ >= (1.0 - self.th_time) * T_:
                is_contact = contact[self.leg_swing]
                self.cnt_contact += is_contact
                self.cnt_contact *= is_contact
                return self.cnt_contact >= self.th_contact or t_ >= (1.0 + self.th_time) * T_
            else:
                self.cnt_contact = 0
                return False
        elif self.gait == "dsp":
            return t_ >= T_

    def start(self, first, footsteps, verbose=True):
        if self.is_start:
            print("please wait for terminating the current footsteps!")
        else:
            # store
            self.leg_swing = first
            self.leg_stance = [key for key in self.leg_pos.keys() if key != first][0]
            self.footsteps = footsteps
            # compute
            footstep = self.footsteps[0]
            ui = 0.5 * sum(self.leg_pos.values())
            ue = self.leg_pos[self.leg_stance].copy()
            un = self.leg_pos[self.leg_swing].copy() + footstep["step"].copy()
            self.set(ui, ue, un, 0.0, self.T_ini, footstep["ssp"])
            # initialize
            self.cnt_footstep = 0
            self.t_elapsed = 0.0
            self.t_phase = 0.0
            self.gait = "dsp"
            self.is_start = True
            if verbose:
                print("start the planned footsteps!\n\t stance={}, swing={}, footsteps={}".format(self.leg_stance, self.leg_swing, self.footsteps))

######################################################
    def step(self, contact):
        self.t_elapsed += self.dt
        self.t_phase += self.dt
        leg_ref = copy.deepcopy(self.leg_pos)
        if self.is_start:
            # update reference com
            dcm_ref = self.trj_dcm(self.t_phase)
            self.com_ref = self.trj_com(self.com_ref, dcm_ref)
            # update reference leg
            if self.gait == "ssp":
                # swing leg
                leg_ini = leg_ref[self.leg_swing]
                if self.cnt_footstep > len(self.footsteps):
                    leg_land = leg_ref[self.leg_stance].copy()
                    leg_land[1] += self.leg_dis[self.leg_swing]
                    T_ = self.T_end
                else:
                    footstep = self.footsteps[self.cnt_footstep-1]
                    leg_land = leg_ini + footstep["step"].copy()
                    T_ = footstep["ssp"]
                leg_ref[self.leg_swing] = self.swing(leg_ini, leg_land, T_, self.t_phase)
                # finish ssp?
                if self.check(contact, T_, self.t_phase):
                    if self.cnt_footstep > len(self.footsteps):
                        # end
                        self.is_start = False
                        self.leg_pos[self.leg_swing] = leg_ref[self.leg_swing].copy()
                        print("terminate the planned footsteps!")
                    else:
                        # to dsp
                        self.gait = "dsp"
                        self.t_phase = 0.0
                        self.leg_stance, self.leg_swing = self.leg_swing, self.leg_stance
                        self.leg_dsp_ini = leg_ref[self.leg_stance]
                        print("{} ssp -> dsp: time={}".format(self.cnt_footstep, self.t_elapsed))
            elif self.gait == "dsp":
                # adjust for inaccurate landing
                if self.cnt_footstep > 0:
                    footstep = self.footsteps[self.cnt_footstep-1]
                    leg_land = leg_ref[self.leg_stance] + footstep["step"].copy()
                    T_ = footstep["dsp"]
                    leg_ref[self.leg_stance] = self.adjust(self.leg_dsp_ini, leg_land)
                else:
                    # move for the first step
                    T_ = self.T_ini
                # finish dsp?
                if self.check(contact, T_, self.t_phase):
                    # to ssp
                    self.gait = "ssp"
                    self.t_phase = 0.0
                    self.leg_pos[self.leg_stance] = leg_ref[self.leg_stance].copy()
                    ui = leg_ref[self.leg_stance].copy()
                    self.cnt_footstep += 1
                    if self.cnt_footstep > len(self.footsteps):
                        # last one step
                        ue = ui.copy()
                        ue[1] += self.leg_dis[self.leg_swing]
                        un = 0.5 * (ue + ui)
                        Ts = self.T_end
                        Td = self.dt
                        Tn = 0.0
                    else:
                        footstep = self.footsteps[self.cnt_footstep-1]
                        ue = leg_ref[self.leg_swing] + footstep["step"].copy()
                        Ts = footstep["ssp"]
                        Td = footstep["dsp"]
                        if self.cnt_footstep == len(self.footsteps):
                            # last one step to allign two feet
                            un = ue.copy()
                            un[1] += self.leg_dis[self.leg_stance]
                            Tn = self.T_end
                        else:
                            # next step
                            nextstep = self.footsteps[self.cnt_footstep]
                            un = ui + nextstep["step"]
                            Tn = nextstep["ssp"]
                    self.set(ui, ue, un, Ts, Td, Tn)
                    print("{} dsp -> ssp: time={}".format(self.cnt_footstep, self.t_elapsed))
        else:
            # move to neutral position (w/ assumption that motion is static)
            ue = 0.5 * sum(leg_ref.values())
            self.com_ref = self.adjust(self.com_ref, ue + self.com_offset)
            for key, lr in leg_ref.items():
                la = ue.copy()
                la[1] = lr[1]
                lr = self.adjust(lr, la)
        # return reference positions
        return self.com_ref.copy(), leg_ref
