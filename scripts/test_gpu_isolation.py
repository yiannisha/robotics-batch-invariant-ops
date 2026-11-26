import unittest

from scripts.gpu_isolation import parse_state,foreign_processes


def xml(processes):return '<nvidia_smi_log><gpu><uuid>GPU-test</uuid><product_name>Test GPU</product_name><processes>'+processes+'</processes></gpu></nvidia_smi_log>'


class Isolation(unittest.TestCase):
    def test_graphics_process_is_not_treated_as_idle(self):
        state=parse_state(xml('<process_info><pid>123</pid><type>G</type><used_memory>435 MiB</used_memory></process_info>'),'GPU-test')
        self.assertEqual([p['pid'] for p in foreign_processes(state)],[123])

    def test_empty_process_inventory_and_explicit_uuid(self):
        self.assertEqual(foreign_processes(parse_state(xml(''),'GPU-test')),[])
        with self.assertRaisesRegex(ValueError,'UUID'):parse_state(xml(''),'GPU-other')

    def test_unsupported_visibility_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'visibility'):parse_state(xml('N/A'),'GPU-test')

    def test_allow_only_benchmark_process_group(self):
        state=dict(processes=[dict(pid=1,type='C'),dict(pid=2,type='G'),dict(pid=3,type='C')])
        other=foreign_processes(state,allowed_pids=[1],allowed_process_group=20,group_resolver=lambda p:{2:20,3:30}[p])
        self.assertEqual([p['pid'] for p in other],[3])


if __name__=='__main__':unittest.main()
