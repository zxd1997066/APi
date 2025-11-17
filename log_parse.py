import argparse
parser = argparse.ArgumentParser(description='Inductor log parser')
parser.add_argument('--log-file', type=str, default="inductor.log", help='log file')
parser.add_argument('--test-list', type=str, default="inductor.log", help='log file')
args = parser.parse_args()

def parse_log(file):
    case_name = []
    results = []
    with open(file, 'r') as reader:
        contents = reader.readlines()
        for line in contents:
            if "test/" in line:
                line1=line.split(" ")[1]
                case_name.append(line1)
            if "PASSED" in line or "SKIPPED" in line or "XFAIL" in line:
                if "PASSED" in line:
                    results.append("PASSED")
                if "SKIPPED" in line:
                    results.append("SKIPPED")
                if "XFAIL" in line:
                    results.append("XFAIL")
    print(len(case_name), len(results))
    return case_name, results

case_name, results = parse_log(args.log_file)

pass_dict = {}
skip_dict = {}
xfail_dict = {}
with open(args.test_list, 'r') as reader:
    contents = reader.readlines()
    for line in contents:
        line=line.replace("\n", "")
        pass_num = 0
        skip_num = 0
        xfail_num = 0
        for i in range(len(case_name)):
            if line in case_name[i]:
                result = results[i]
                if result == "PASSED":
                    pass_num = pass_num + 1
                if result == "SKIPPED":
                    skip_num = skip_num + 1
                if result == "XFAIL":
                    xfail_num = xfail_num + 1
        pass_dict[line] = pass_num
        skip_dict[line] = skip_num
        xfail_dict[line] = xfail_num
pass_dict = str(pass_dict).replace("'", '"').replace(',', ',\n')
skip_dict = str(skip_dict).replace("'", '"').replace(',', ',\n')
xfail_dict = str(xfail_dict).replace("'", '"').replace(',', ',\n')
print(xfail_dict)