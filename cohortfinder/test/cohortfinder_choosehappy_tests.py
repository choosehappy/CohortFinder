import unittest
from ..cohortfinder import runCohortFinder
from argparse import Namespace

class TestCohortFinder(unittest.TestCase):

    def test_cohortfinder(self):
        # add argparse arguments for cohortfinder here.
        args = Namespace()
        self.assertEqual(runCohortFinder(args), "CohortFinder is running")

