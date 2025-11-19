import argparse
parser = argparse.ArgumentParser(description='Inductor log parser')
parser.add_argument('--log-file-cuda', type=str, default="inductor.log", help='log file')
parser.add_argument('--log-file-xpu', type=str, default="inductor.log", help='log file')
# parser.add_argument('--test-list', type=str, default="inductor.log", help='log file')
args = parser.parse_args()

def parse_log(file, device):
    case_name = []
    results = []
    result_dict = {}
    with open(file, 'r') as reader:
        contents = reader.readlines()
        for line in contents:
            if "test/" in line:
                if device == "cuda":
                    line1=line.split(" ")[1]
                else:
                    line1=line.split(" ")[0]
                case_name.append(line1)
            if "PASSED" in line or "SKIPPED" in line or "XFAIL" in line or "FAILED":
                if "PASSED" in line:
                    results.append("PASSED")
                    result_dict[line1] = "PASSED"
                elif "SKIPPED" in line:
                    results.append("SKIPPED")
                    result_dict[line1] = "SKIPPED"
                elif "XFAIL" in line:
                    results.append("XFAIL")
                    result_dict[line1] = "XFAIL"
                elif "FAILED" in line:
                    results.append("FAILED")
                    result_dict[line1] = "FAILED"
                 
    print(len(case_name), len(results))
    return case_name, results, result_dict

case_name_cuda, results_cuda, dict_cuda = parse_log(args.log_file_cuda, "cuda")
case_name_xpu, results_xpu, dict_xpu = parse_log(args.log_file_xpu, "xpu")
case_name_total = list(dict.fromkeys(case_name_cuda + case_name_xpu))
dict_total = {}
for i in range(len(case_name_total)):
    if case_name_total[i] in case_name_xpu:
        result_xpu = dict_xpu[case_name_total[i]]
    else:
        result_xpu = "missed"
    if case_name_total[i] in case_name_cuda:
        result_cuda = dict_cuda[case_name_total[i]]
    else:
        result_cuda = "missed"
    dict_total[case_name_total[i]] = [result_xpu+" "+result_cuda]

dict_total = str(dict_total).replace("'", '"').replace(',', ',\n')
print(dict_total)
    
# pass_dict = {}
# skip_dict = {}
# xfail_dict = {}
# with open(args.test_list, 'r') as reader:
#     contents = reader.readlines()
#     for line in contents:
#         line=line.replace("\n", "")
#         pass_num = 0
#         skip_num = 0
#         xfail_num = 0
#         for i in range(len(case_name)):
#             if line in case_name[i]:
#                 result = results[i]
#                 if result == "PASSED":
#                     pass_num = pass_num + 1
#                 if result == "SKIPPED":
#                     skip_num = skip_num + 1
#                 if result == "XFAIL":
#                     xfail_num = xfail_num + 1
#         pass_dict[line] = pass_num
#         skip_dict[line] = skip_num
#         xfail_dict[line] = xfail_num
# pass_dict = str(pass_dict).replace("'", '"').replace(',', ',\n')
# skip_dict = str(skip_dict).replace("'", '"').replace(',', ',\n')
# xfail_dict = str(xfail_dict).replace("'", '"').replace(',', ',\n')
# print(xfail_dict)