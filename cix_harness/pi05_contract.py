"""Explicit LIBERO OSC_POSE -> bounded Baxter bridge adaptation.

LIBERO/robosuite 1.4.1 uses normalized Cartesian deltas, axis-angle rotation,
and gripper +1=close/-1=open. These are not Bridge/OpenVLA's physical units.
"""
import math

from .contract import action7, vector


def quaternion(values):
    q = vector(values, 4, 'quaternion')
    if abs(sum(x*x for x in q) - 1) > 1e-4:
        raise ValueError('Quaternion must have unit norm')
    return q


def multiply(a, b):
    x, y, z, w = a
    u, v, t, s = b
    return [w*u+x*s+y*t-z*v, w*v-x*t+y*s+z*u,
            w*t+x*v-y*u+z*s, w*s-x*u-y*v-z*t]


def rotate(q, xyz):
    return multiply(multiply(q, list(xyz)+[0]), [-q[0], -q[1], -q[2], q[3]])[:3]


def tcp_offset(profile):
    """Measured SDK-endpoint -> grasp-point offset in the SDK tool frame."""
    return vector(profile.get('tcp_offset_tool_m', [0, 0, 0]), 3, 'TCP offset')


def tcp_position(position, orientation, offset):
    position = vector(position, 3, 'endpoint position')
    shift = rotate(quaternion(orientation), vector(offset, 3, 'TCP offset'))
    return [a+b for a,b in zip(position, shift)]


def endpoint_for_tcp_delta(position, orientation, target_orientation, delta, offset):
    """Keep the requested TCP delta while rotating the offset tool."""
    start = tcp_position(position, orientation, offset)
    shift = rotate(quaternion(target_orientation), vector(offset,3,'TCP offset'))
    return [a+b-c for a,b,c in zip(start,vector(delta,3,'TCP delta'),shift)]


def rotvec(q):
    q = quaternion(q)
    # LIBERO/robosuite quat2axisangle preserves angles above pi. Shortest-arc
    # canonicalization would flip the trained ~+pi x component to ~-pi.
    n = math.sqrt(sum(x*x for x in q[:3]))
    if n < 1e-10:
        return [2*x for x in q[:3]]
    angle = 2*math.atan2(n, q[3])
    return [x*angle/n for x in q[:3]]


def state8(state, profile):
    q = quaternion(profile['policy_from_base_xyzw'])
    translation = vector(profile['policy_from_base_translation_m'], 3, 'policy origin')
    position = rotate(q, tcp_position(state['eef_m'], state['eef_xyzw'], tcp_offset(profile)))
    orientation = multiply(multiply(q, quaternion(state['eef_xyzw'])),
                           quaternion(profile['tool_frame_offset_xyzw']))
    reference=profile.get('orientation_quaternion_reference_xyzw')
    if reference is not None and sum(a*b for a,b in zip(orientation,quaternion(reference))) < 0:
        orientation=[-v for v in orientation]
    aperture = vector([state['gripper_percent']], 1, 'measured gripper aperture')[0]
    if not 0 <= aperture <= 100:
        raise ValueError('Gripper aperture outside 0..100')
    # This is an explicit cross-embodiment feature mapping, not measured Panda joints.
    half_open = 0.04 * aperture / 100
    return [a+b for a,b in zip(position, translation)] + rotvec(orientation) + [half_open, -half_open]


def physical_delta(raw):
    values = vector(raw, 7, 'LIBERO OSC action')
    # Match robosuite's controller input clipping and physical output range.
    values = [max(-1.0, min(1.0, x)) for x in values]
    r = [x*0.5 for x in values[3:6]]
    angle = math.sqrt(sum(x*x for x in r))
    scale = math.sin(angle/2)/angle if angle > 1e-10 else 0.5
    x, y, z = [v*scale for v in r]
    w = math.cos(angle/2)
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return action7([x*0.05 for x in values[:3]] + [roll, pitch, yaw, 0.0 if values[6] > 0 else 1.0])
