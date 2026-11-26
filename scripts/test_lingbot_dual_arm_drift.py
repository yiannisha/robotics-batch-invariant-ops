import unittest
import numpy as np
from scripts.measure_lingbot_pilot_stability import dual_arm_drift


class DualArmDrift(unittest.TestCase):
    def poses(self):
        value=np.zeros((3,2,7));value[:,:,3]=1.;return value

    def test_translation_is_reported_separately_per_arm(self):
        x=self.poses();y=x.copy();y[:,0,0]=0.01;r=dual_arm_drift(x,y)
        self.assertAlmostEqual(r['arms']['left']['position_max_m'],0.01)
        self.assertEqual(r['arms']['right']['position_max_m'],0.)

    def test_quaternion_sign_and_known_rotation(self):
        x=self.poses();y=x.copy();y[:,0,3:]*=-1
        y[:,1,3]=np.sqrt(.5);y[:,1,4]=np.sqrt(.5);r=dual_arm_drift(x,y)
        self.assertEqual(r['arms']['left']['rotation_max_rad'],0.)
        self.assertAlmostEqual(r['arms']['right']['rotation_max_rad'],np.pi/2)

    def test_common_horizon_excludes_unshared_tail(self):
        x=self.poses();y=x.copy();y[-1,:,0]=100
        r=dual_arm_drift(x[:2],y);self.assertEqual(r['common_frames'],2)
        self.assertEqual(r['arms']['left']['position_max_m'],0.)


if __name__=='__main__':unittest.main()
