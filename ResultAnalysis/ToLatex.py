import numpy as np





def matrix_to_latex(mean, std, rst, col_names, row_names, oder='min'):
    num_cols = len(mean.keys())
    num_rows = len(row_names)

    if len(col_names) != num_cols or len(row_names) != num_rows:
        raise ValueError("Mismatch between matrix dimensions and provided row/column names.")

    latex_code = []

    latex_code.append("\\begin{table*}[t!]")
    latex_code.append("    \\scriptsize")
    latex_code.append("    \\centering")
    latex_code.append("    \\caption{Performance comparisons of the quality of solutions obtained by different algorithms.}")
    latex_code.append("    \\label{tab:rq1_result}%")
    latex_code.append("    \\resizebox{1.0\\textwidth}{!}{")
    latex_code.append("    \\begin{tabular}{c|" + "".join(["c"] * (num_rows)) + "}")
    latex_code.append("        \\hline")

    # Adding column names
    col_header = " & ".join([""] + row_names) + " \\\\"
    latex_code.append("        " + col_header)
    latex_code.append("        \\hline")

    # Adding rows
    for i in range(num_cols):
        str_data = []
        for j in range(num_rows):
            str_format = ""
            if oder =='min':
                if mean[col_names[i]][j] == np.min(mean[col_names[i]]):
                    str_format +="\cellcolor[rgb]{ .682,  .667,  .667}\\textbf{"
                    str_format += "%.3E(%.3E)" % (float(mean[col_names[i]][j]), std[col_names[i]][j])
                    str_format += "}"
                    str_data.append(str_format)
                else:
                    if rst[col_names[i]][row_names[j]] == '+':
                        str_data.append("%.3E(%.3E)$^\dagger$" % (float(mean[col_names[i]][j]), std[col_names[i]][j]))
                    else:
                        str_data.append("%.3E(%.3E)" % (float(mean[col_names[i]][j]), std[col_names[i]][j]))
            else:
                if mean[col_names[i]][j] == np.max(mean[col_names[i]]):
                    str_format +="\cellcolor[rgb]{ .682,  .667,  .667}\\textbf{"
                    str_format += "%.3E(%.3E)" % (float(mean[col_names[i]][j]), std[col_names[i]][j])
                    str_format += "}"
                    str_data.append(str_format)
                else:
                    if rst[col_names[i]][row_names[j]] == '+':
                        str_data.append("%.3E(%.3E)$^\dagger$" % (float(mean[col_names[i]][j]), std[col_names[i]][j]))
                    else:
                        str_data.append("%.3E(%.3E)" % (float(mean[col_names[i]][j]), std[col_names[i]][j]))
        test_name = col_names[i].split('_')[0] + col_names[i].split('_')[1]
        row_data = " & ".join(["\\texttt{" + f'{test_name}' + "}"] + str_data) + " \\\\"
        latex_code.append("        " + row_data)

    latex_code.append("        \\hline")
    latex_code.append("    \\end{tabular}")
    latex_code.append("    }")
    latex_code.append("    \\begin{tablenotes}")
    latex_code.append("        \\tiny")
    latex_code.append("        \\item The labels in the first column are the combination of the first letter of test problem and the number of variables, e.g., A4 is Ackley problem with $n=4$.")
    latex_code.append("        \\item $^\\dagger$ indicates that the best algorithm is significantly better than the other one according to the Wilcoxon signed-rank test at a 5\\% significance level.")
    latex_code.append("    \\end{tablenotes}")
    latex_code.append("\\end{table*}%")

    return "\n".join(latex_code)
