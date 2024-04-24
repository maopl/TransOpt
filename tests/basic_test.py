import unittest
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.insert(0, package_dir)

from benchmark.instantiate_problems import get_testsuits
from optimizer.construct_optimizer import get_optimizer
from transopt.KnowledgeBase.kb_builder import construct_knowledgebase
from transopt.KnowledgeBase.TransferDataHandler import OptTaskDataHandler


class TestTransOptInstallation(unittest.TestCase):
    def test_imports(self):
        self.assertIsNotNone(get_testsuits)
        self.assertIsNotNone(get_optimizer)
        self.assertIsNotNone(construct_knowledgebase)
        self.assertIsNotNone(OptTaskDataHandler)

    def test_basic_operations(self):
        args = argparse.Namespace(
            init_method="random",
            init_number=7,
            exp_path=f"{package_dir}/../LFL_experiments",
            exp_name="test",
            seed=0,
            optimizer="ParEGO",
            verbose=True,
            normalize="norm",
            source_num=2,
            selector="None",
            save_mode=1,
            load_mode=False,
            acquisition_func="LCB",
        )

        kb = construct_knowledgebase(args)
        self.assertIsNotNone(kb)

        testsuits = get_testsuits({"DBMS": {"budget": 11, "time_stamp": 3}}, args)
        self.assertIsNotNone(testsuits)

        optimizer = get_optimizer(args)
        self.assertIsNotNone(optimizer)

        data_handler = OptTaskDataHandler(kb, args)
        try:
            optimizer.optimize(testsuits, data_handler)
        except Exception as e:
            self.fail(f"Optimizer.optimize failed with an exception: {e}")


if __name__ == "__main__":
    unittest.main()
