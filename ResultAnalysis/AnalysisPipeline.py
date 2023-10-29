from ResultAnalysis.PlotAnalysis import plot_registry
from ResultAnalysis.TableAnalysis import table_registry
from ResultAnalysis.AnalysisBase import AnalysisBase
from ResultAnalysis.AnalysisReport import create_report



def analysis_pipeline(Exper_folder, tasks, methods, seeds):
    ab = AnalysisBase(Exper_folder, tasks=tasks,methods= methods,seeds= seeds)
    ab.read_data_from_kb()
    Exper_folder = Exper_folder / 'analysis'
    for plot_name, plot_func in plot_registry.items():
        plot_func(ab, Exper_folder)  # 假设你的度量函数需要额外的参数

    for table_name, table_func in table_registry.items():
        table_func(ab, Exper_folder)  # 假设你的度量函数需要额外的参数

    create_report(Exper_folder)

