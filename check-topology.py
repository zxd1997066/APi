import os
import sys

# Get the xelink group card affinity
ret = os.system("xpu-smi topology -m 2>&1|tee topology.log")
if ret == 0:
    gpu_dict = {}
    with open("topology.log") as file:
        lines = file.readlines()
        for line in lines:
            if "CPU Affinity" in line:
                continue
            line = line.strip()
            if line.startswith("GPU "):
                items = line.split(" ")
                items = [x for x in items if x]
                gpu_id = items[1]
                cpu_affinity = items[-1].split(",")[0]
                start_cpu = cpu_affinity.split("-")[0]
                end_cpu = cpu_affinity.split("-")[1]
                print(start_cpu, end_cpu)
                i = gpu_id.split("/")[0]
                affinity = ""
                for j, item in enumerate(items):
                    if "SYS" not in item and ("XL" in item or "S" in item):
                        if len(affinity) == 0:
                            affinity = str(j - 2)
                        else:
                            affinity = affinity + "," + str(j - 2)
                gpu_dict[i] = affinity
        print(gpu_dict)

    max_affinity = ""
    value_to_keys = {}
    for key, value in gpu_dict.items():
        if value not in value_to_keys:
            value_to_keys[value] = []
        value_to_keys[value].append(key)
    for key, value in value_to_keys.items():
        print(','.join(value_to_keys[key]))

    os.environ["ZE_AFFINITY_MASK"] = str(max_affinity)
    print(str("ZE_AFFINITY_MASK=" + os.environ.get("ZE_AFFINITY_MASK")))

else:
    print("xpu-smi topology failed")

    sys.exit(255)






