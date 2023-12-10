# Import all the benchmark classes here
from transopt.Benchmark.CSSTuning.Compiler import GCC, LLVM
from transopt.Benchmark.CSSTuning.DBMS import DBMSTuning
from transopt.Benchmark.HPO.HPORes18 import HPOResNet
from transopt.Benchmark.HPO.HPOSVM import SupportVectorMachine
from transopt.Benchmark.HPO.HPOXGBoost import XGBoostBenchmark
from transopt.Benchmark.RL.LunarlanderBenchmark import LunarlanderBenchmark
from transopt.Benchmark.Synthetic.MovingPeakBenchmark import MovingPeakBenchmark
from transopt.Benchmark.Synthetic.MultiObjBenchmark import AckleySphereOptBenchmark
from transopt.Benchmark.Synthetic.SyntheticBenchmark import (
    SphereOptBenchmark,
    RastriginOptBenchmark,
    SchwefelOptBenchmark,
    LevyROptBenchmark,
    GriewankOptBenchmark,
    RosenbrockOptBenchmark,
    DropwaveROptBenchmark,
    LangermannOptBenchmark,
    RotatedHyperEllipsoidOptBenchmark,
    SumOfDifferentPowersOptBenchmark,
    StyblinskiTangOptBenchmark,
    PowellOptBenchmark,
    DixonPriceOptBenchmark,
    cpOptBenchmark,
    mpbOptBenchmark,
    AckleyOptBenchmark,
    EllipsoidOptBenchmark,
    DiscusOptBenchmark,
    BentCigarOptBenchmark,
    SharpRidgeOptBenchmark,
    GriewankRosenbrockOptBenchmark,
    KatsuuraOptBenchmark,
)
