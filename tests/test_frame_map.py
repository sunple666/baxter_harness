import math
import unittest
from cix_harness.frame_map import build
from cix_harness.pi05_contract import rotate,state8


class FrameMapTest(unittest.TestCase):
    def references(self):
        return {'reference_definition':'Synthetic correspondences for tests only',
            'base_tcp_points_m':[[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],
            'policy_points_m':[[1,2,3],[1,2.1,3],[.9,2,3],[1,2,3.1]],
            'sdk_reference_xyzw':[1,0,0,0],'policy_reference_xyzw':[1,0,0,0],
            'tcp_offset_tool_m':[0,0,.0126658307480536]}

    def test_rigid_transform_inverse_and_tool_reference(self):
        result=build(self.references());p=result['policy_profile']
        self.assertFalse(p['commissioned']);self.assertEqual(result['extra_validation_points'],1)
        for error in result['point_residuals_m']:self.assertLess(error,1e-12)
        d=[.03,-.01,.02]
        actual=rotate(result['bridge_candidate']['policy_to_base_xyzw'],rotate(p['policy_from_base_xyzw'],d))
        for a,b in zip(actual,d):self.assertAlmostEqual(a,b)
        state={'eef_m':[0,0,0],'eef_xyzw':[1,0,0,0],'gripper_percent':50}
        output=state8(state,p)
        self.assertAlmostEqual(output[3],math.pi)
        self.assertAlmostEqual(output[4],0)
        self.assertAlmostEqual(output[5],0)

    def test_nonrigid_or_reflected_extra_point_is_rejected(self):
        for point in [[1,2,2.9],[1,2,3.3]]:
            d=self.references();d['policy_points_m'][3]=point
            with self.assertRaisesRegex(ValueError,'residual'):build(d)

    def test_degenerate_and_invalid_data_rejected(self):
        for key,value in [('base_tcp_points_m',[[0,0,0],[.1,0,0],[.2,0,0]]),
                          ('max_residual_m',float('nan')),('tcp_offset_tool_m',[0,0,float('inf')])]:
            d=self.references();d[key]=value
            with self.assertRaises(ValueError):build(d)
