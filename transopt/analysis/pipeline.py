from pathlib import Path
from transopt.analysis.analysisbase import AnalysisBase
from transopt.analysis.comparisonbase import ComparisonBase
from transopt.agent.registry import analysis_registry, comparison_registry
from transopt.analysis.plot import *
from transopt.analysis.statistics import *

# PlotAnalysis import plot_registry
# from transopt.ResultAnalysis.TableAnalysis import table_registry
# from transopt.ResultAnalysis.TrackOptimization import track_registry
# from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
# from transopt.ResultAnalysis.AnalysisReport import create_report


def analysis(Exper_folder, datasets, data_manager, args):
    ab = AnalysisBase(Exper_folder, datasets, data_manager)
    ab.read_data_from_db()
    Exp_folder = Path(Exper_folder) / 'analysis'
    
    for name in analysis_registry.list_names():
        func = analysis_registry.get(name)
        func(ab, Exp_folder)
    

def comparison(Exper_folder, datasets, data_manager, args):
    cb = ComparisonBase(Exper_folder, datasets, data_manager)
    cb.read_data_from_db()
    Exp_folder = Path(Exper_folder) / 'comparison'
    
    for name in comparison_registry.list_names():
        func = comparison_registry.get(name)
        func(cb, Exp_folder) 



