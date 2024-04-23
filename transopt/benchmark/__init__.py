# # Import all the benchmark classes here
# from transopt.benchmark.CSSTuning.Compiler import GCCTuning, LLVMTuning
# from transopt.benchmark.CSSTuning.DBMS import DBMSTuning
# from transopt.benchmark.HPO.HPORes18 import HPOResNet
# from transopt.benchmark.HPO.HPOSVM import SupportVectorMachine
# from transopt.benchmark.HPO.HPOXGBoost import XGBoostBenchmark
# from transopt.benchmark.RL.LunarlanderBenchmark import LunarlanderBenchmark
# from transopt.benchmark.Synthetic.MovingPeakBenchmark import MovingPeakBenchmark
# from transopt.benchmark.Synthetic.MultiObjBenchmark import AckleySphereOptBenchmark
from transopt.benchmark.problem_base.base import ProblemBase
from benchmark.problem_base.non_tab_problem import NonTabularProblem
from benchmark.problem_base.tab_problem import TabularProblem
from benchmark.problem_base.transfer_problem import TransferProblem, RemoteTransferOptBenchmark
from transopt.benchmark.Synthetic.SyntheticBenchmark import (
    # SphereOptBenchmark,
    # RastriginOptBenchmark,
    # SchwefelOptBenchmark,
    # LevyROptBenchmark,
    # GriewankOptBenchmark,
    # RosenbrockOptBenchmark,
    # DropwaveROptBenchmark,
    # LangermannOptBenchmark,
    # RotatedHyperEllipsoidOptBenchmark,
    # SumOfDifferentPowersOptBenchmark,
    # StyblinskiTangOptBenchmark,
    # PowellOptBenchmark,
    # DixonPriceOptBenchmark,
    # cpOptBenchmark,
    # mpbOptBenchmark,
    Ackley,
    # EllipsoidOptBenchmark,
    # DiscusOptBenchmark,
    # BentCigarOptBenchmark,
    # SharpRidgeOptBenchmark,
    # GriewankRosenbrockOptBenchmark,
    # KatsuuraOptBenchmark,
)

from benchmark.instantiate_problems import InstantiateProblems
