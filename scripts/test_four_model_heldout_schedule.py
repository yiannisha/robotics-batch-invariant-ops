import copy
import unittest

from scripts.audit_four_model_heldout import schedule


class HeldoutSchedule(unittest.TestCase):
    def setUp(self):
        self.observations=[dict(split='heldout',request_ids=[str(i)]) for i in range(4)]
        self.source=dict(companion_replay_indices=[1,2,3])
        self.row=dict(logical_batch_size=4,target_batch_index=2,request_id='0',companion_type='diverse',
            transformation='batch_position',request_ordering=['1','2','0','3'],companion_ids=['1','2','3'],batch_size=4)

    def test_batch_position(self):schedule(self.row,self.source,self.observations,0)

    def test_partition_uses_physical_half_batch(self):
        r={**self.row,'transformation':'partition','target_batch_index':0,'request_ordering':['0','1','2','3'],'batch_size':2}
        schedule(r,self.source,self.observations,0)
        r['batch_size']=4
        with self.assertRaisesRegex(ValueError,'Physical batch'):schedule(r,self.source,self.observations,0)

    def test_permutation_and_duplicate_inputs(self):
        r={**self.row,'transformation':'permutation','target_batch_index':3,'request_ordering':['3','2','1','0'],'companion_ids':['3','2','1']}
        schedule(r,self.source,self.observations,0)
        r={**self.row,'companion_type':'duplicate','request_ordering':['0']*4,'companion_ids':['0']*3}
        schedule(r,self.source,self.observations,0)

    def test_cross_split_companion_rejected(self):
        self.observations[1]['split']='diagnostic'
        with self.assertRaisesRegex(ValueError,'crosses heldout'):schedule(self.row,self.source,self.observations,0)

    def test_changed_order_and_target_rejected(self):
        r=copy.deepcopy(self.row);r['request_ordering']=['0','1','2','3']
        with self.assertRaisesRegex(ValueError,'ordering'):schedule(r,self.source,self.observations,0)
        r=copy.deepcopy(self.row);r['target_batch_index']=4
        with self.assertRaisesRegex(ValueError,'target'):schedule(r,self.source,self.observations,0)


if __name__=='__main__':unittest.main()
