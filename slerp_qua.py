"""
contains functions interpolating extrinsics

SLERP taken from: https://github.com/apple/ml-neuman/blob/eaa4665908ba1e39de5f40ef1084503d1b6ab1cb/geometry/transformations.py
"""
import numpy as np
import math
# from colmap_find_scale.read_write_model import qvec2rotmat, rotmat2qvec
from collections import namedtuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import warnings

_EPS = np.finfo(float).eps * 4.0
Rmtx = namedtuple("Rmtx", ["flat"])

class CameraSpline:
    def __init__(self, ts, w2cs, coords):
        """
        ts (list: float/ints): location of w2cs
        coords (list: array): location of camera in 3d space
        """
        self.ts = ts
        self.w2cs = w2cs
        self.coords = coords

        if len(self.ts) != len(self.w2cs):
            warnings.warn(f"number of triggers {len(self.ts)} != num cameras {len(self.w2cs)}, assume extra cameras are not in triggers")
            min_size = min(len(self.w2cs), len(self.ts))
            self.w2cs, self.coords = self.w2cs[:min_size], self.coords[:min_size]
            self.ts = self.ts[:min_size]
        
        self.rot_interpolator = Slerp(self.ts, Rotation.from_matrix(self.w2cs))
        self.trans_interpolator = interp1d(x=self.ts, y=self.coords, axis=0, kind="linear", bounds_error=True)
    

    def interpolate(self, t):
        t = np.clip(t, self.ts[0], self.ts[-1])

        return self.trans_interpolator(t), self.rot_interpolator(t).as_matrix()

    def interp_as_mtx(self, t):
        T, R = self.interpolate(t)

        if len(R.shape) == 2:
            return np.concatenate([R, T.reshape(3,1)], axis=1)

        if len(R.shape) == 3:
            return np.concatenate([R, T.reshape(len(R),3,1)], axis = -1)
        
        assert 0, "CODE SHOULD NOT GET HERE"
    
def lanczos_kernel(x, a=2):
    return np.sinc(x) * np.sinc(x / a) * (np.abs(x) < a)


def q_to_exp(q, q0):
    bs = q.shape[:-1]
    r = Rotation.from_quat(q.reshape(-1, 4))
    r0 = Rotation.from_quat(q0.reshape(-1, 4))

    rp = r * r0.inv()

    qp = rp.as_quat().reshape(*bs, 4)

    axis = qp[..., :3]
    axlen = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / (axlen + 1e-7)
    theta = np.arccos(qp[..., 3:4]) * 2
    theta -= 2 * np.pi * (theta > np.pi)

    return theta * axis

def exp_to_q(ep, q0):
    bs = ep.shape[:-1]
    rp = Rotation.from_rotvec(ep.reshape(-1, 3))
    r0 = Rotation.from_quat(q0.reshape(-1, 4))
    r = rp * r0

    q = r.as_quat().reshape(*bs, 4)
    return q



class LanczosSpline:
    def __init__(self, ts, w2cs, coords) -> None:
        self.orig_ts, self.w2cs, self.coords = ts, w2cs, coords
        self.t_base = self.orig_ts[0]
        self.orig_ts = self.orig_ts - self.t_base

        rots = Rotation.from_matrix(self.w2cs)
        assert np.abs(Rotation.from_quat(rots.as_quat()).as_matrix() - self.w2cs).sum() < 1e-6, "is mirror transform!"

        if len(self.orig_ts) != len(self.w2cs):
            warnings.warn(f"number of triggers {len(self.orig_ts)} != num cameras {len(self.w2cs)}, assume extra cameras are not in triggers")
            min_size = min(len(self.w2cs), len(self.orig_ts))
            self.w2cs, self.coords = self.w2cs[:min_size], self.coords[:min_size]
            self.orig_ts = self.orig_ts[:min_size]
        
        self.fix_assumption()
        rots = Rotation.from_matrix(self.w2cs)
        self.q_orig = rots.as_quat()
        self.slerp_orig = Slerp(self.orig_ts, rots)
    
    
    def fix_assumption(self):
        """
        Lanczos assumes dt interval is same for all, after fixing "yes trigger, no frames"; assumption is broken.
        This function will inject an average position and make a new t to fix it
        """

        dt = self.orig_ts[1] - self.orig_ts[0]

        new_ts, new_w2cs, new_coords = [self.orig_ts[0]], [self.w2cs[0]], [self.coords[0]]
        for i in range(len(self.orig_ts) - 1):
            curr_dt = np.abs(self.orig_ts[i+1] - self.orig_ts[i])
            if curr_dt > dt + 50:
                # Number of steps to interpolate
                steps = int(np.ceil(curr_dt / dt))

                for step in range(1, steps):
                    # Interpolate time
                    t = self.orig_ts[i] + step * dt
                    new_ts.append(t)

                    # Linearly interpolate w2c and coords
                    interp_w2c = self.w2cs[i] + (self.w2cs[i+1] - self.w2cs[i]) * (step / steps)
                    interp_coord = self.coords[i] + (self.coords[i+1] - self.coords[i]) * (step / steps)

                    new_w2cs.append(interp_w2c)
                    new_coords.append(interp_coord)

            # Add the original next timestamp, w2c, and coord
            new_ts.append(self.orig_ts[i+1])
            new_w2cs.append(self.w2cs[i+1])
            new_coords.append(self.coords[i+1])

        # Update the class attributes
        self.orig_ts = np.array(new_ts)
        self.w2cs = np.stack(new_w2cs)
        self.coords = np.stack(new_coords)


    
    def interp_trans(self, ts, a=4):
        dx_orig = self.orig_ts[1] - self.orig_ts[0]
        i0 = np.floor(ts / dx_orig).astype(int)
        i_off = np.arange(1 - a, a + 1)
        i_all = i0[:, None] + i_off[None, :]
        i_all = np.clip(i_all, 0, len(self.orig_ts) - 1)
        y_all = self.coords[i_all]
        t = (ts / dx_orig) % 1.0
        t_all = t[:, None] - i_off[None, :]
        w_all = lanczos_kernel(t_all, a)
        return np.sum(y_all * w_all[..., None], axis=1)


    def interp_rot(self, ts, a=4, iterations=100):
        dt_orig = self.orig_ts[1] - self.orig_ts[0]
        i0 = np.floor(ts / dt_orig).astype(np.int32)
        i_off = np.arange(1 - a, a + 1)
        i_all = i0[:, None] + i_off[None, :]
        i_all = np.clip(i_all, 0, len(self.orig_ts) - 1)

        u = (ts / dt_orig) % 1.0
        u_all = u[:, None] - i_off[None, :]
        w_all = lanczos_kernel(u_all, a)

        q_all = self.q_orig[i_all]
        r_init = self.slerp_orig(ts)
        q_init = r_init.as_quat()
        q_inter = q_init

        for j in range(iterations):
            e_all = q_to_exp(q_all, q_inter[..., None, :] + q_all * 0.0)
            theta = np.linalg.norm(e_all, axis=-1, keepdims=True)
            w_theta = (1 + np.cos(theta))**(1/2)
            e_all = (e_all * w_theta) / np.sum(w_theta, axis=-2, keepdims=True)
            e_new = np.sum(w_all[..., None] * e_all, axis=-2) * 0.1
            q_new = exp_to_q(e_new, q_inter)
            q_inter = q_new
        
        return q_inter


    def interpolate(self, ts):
        ts = ts - self.t_base
        ts = np.clip(ts, self.orig_ts[0], self.orig_ts[-1])

        new_trans = self.interp_trans(ts)
        q_rot = self.interp_rot(ts)
        mtx_rot = Rotation.from_quat(q_rot).as_matrix()

        return new_trans, mtx_rot




def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def create_interpolated_cams(interp_ts, ctrl_ts, ctrl_extrns):
    """
    input:
        interp_ts (np.array): starting times at which the image is accumulated
        ctrl_ts (np.array): col img starting time
        ctrl_extrns (np.array): world to camera matrix extrinsics of event camera at trigger times

    returns:
        ecams_int (np.array): interpolated extrinsic positions
    """

    def split_extrnx(w2cs):
        Rs = w2cs[:,:3,:3]
        ts = w2cs[:,:3, 3]
        return Rs, ts
    
    Rs, ts = split_extrnx(ctrl_extrns)
    cam_spline = CameraSpline(ctrl_ts, Rs, ts)
    # cam_spline = LanczosSpline(triggers, Rs, ts)
    int_ts, int_Rs =  cam_spline.interpolate(interp_ts)

    if len(int_ts.shape) == 2:
        int_ts = int_ts[..., None]
    
    int_cams = np.concatenate([int_Rs, int_ts], axis=-1)
    bot = np.zeros((4,))
    bot[-1] = 1
    bot = bot[None, None]
    int_cams = np.concatenate([int_cams, np.concatenate([bot]*len(int_cams))], axis = -2)
    return int_cams