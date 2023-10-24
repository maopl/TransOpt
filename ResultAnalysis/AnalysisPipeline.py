from ResultAnalysis.PlotAnalysis import plot_registry
from ResultAnalysis.TableAnalysis import table_registry
from ResultAnalysis.AnalysisBase import AnalysisBase




def analysis_pipeline(Exper_folder, Methods_list, Seeds):
    ab = AnalysisBase(Exper_folder, Methods_list, Seeds)
    ab.read_data_from_kb()
    Exper_folder = Exper_folder / 'analysis'
    for plot_name, plot_func in plot_registry.items():
        save_path = Exper_folder / f'{plot_name}'
        plot_func(ab.results, save_path)  # 假设你的度量函数需要额外的参数

    for table_name, table_func in table_registry.items():
        save_path = Exper_folder / f'{table_name}'
        table_func(ab.results, save_path)  # 假设你的度量函数需要额外的参数




