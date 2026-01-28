import argparse
parser = argparse.ArgumentParser(description='Inductor log parser')
parser.add_argument('--log-file', type=str, default="inductor.log", help='log file')
args = parser.parse_args()

def parse_log(file):
    case_name = []
    results = []
    file_names = []
    with open(file, 'r') as reader:
        contents = reader.readlines()
        Flag = True
        for line in contents:
            if "test/" in line:
                line1=line.split(" ")[1]
                file_name = line1.split(":")[0]
                if Flag:
                    case_name.append(line1)
                    Flag = False
                if file_name not in file_names:
                    file_names.append(file_name)
            if "mPASSED" in line or "mSKIPPED" in line or "mFAILED" in line:
                if "mPASSED" in line:
                    results.append("PASSED")
                if "mSKIPPED" in line:
                    results.append("SKIPPED")
                if "mFAILED" in line:
                    results.append("FAILED")
                Flag = True
    print(len(case_name), len(results))
    return case_name, results, file_names

case_name, results, file_names = parse_log(args.log_file)

pass_dict = {}
skip_dict = {}
fail_dict = {}
for name in file_names:
    pass_num = 0
    skip_num = 0
    fail_num = 0
    for i in range(len(case_name)):
        if name in case_name[i]:
            result = results[i]
            if result == "PASSED":
                pass_num = pass_num + 1
            if result == "SKIPPED":
                skip_num = skip_num + 1
            if result == "FAILED":
                fail_num = fail_num + 1
    pass_dict[name] = pass_num
    skip_dict[name] = skip_num
    fail_dict[name] = fail_num
pass_dict = str(pass_dict).replace("'", '"').replace(',', ',\n')
skip_dict = str(skip_dict).replace("'", '"').replace(',', ',\n')
fail_dict = str(fail_dict).replace("'", '"').replace(',', ',\n')
print(fail_dict)