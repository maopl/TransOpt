from ResultAnalysis.Metrics import metric_registry
from ResultAnalysis.AnalysisBase import AnalysisBase




def analysis_pipeline(Exper_folder, Methods_list, Seeds):
    ab = AnalysisBase(Exper_folder, Methods_list, Seeds)
    ab.read_data_from_kb()
    Exper_folder = Exper_folder / 'analysis'
    for metric_name, metric_func in metric_registry.items():
        save_path = Exper_folder / f'{metric_name}'
        metric_func(ab.results, save_path)  # 假设你的度量函数需要额外的参数




