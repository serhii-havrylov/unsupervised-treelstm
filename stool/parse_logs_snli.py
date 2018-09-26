import glob
import numpy as np
from collections import defaultdict


results = defaultdict(list)


for log_file_path in glob.glob("data/snli_detailed_grid_exp/logs/*.log"):
# for log_file_path in glob.glob("data/snli_grid_exp/logs/*.log"):
    with open(log_file_path) as file:
        hyper_params = None
        key = None
        best_epoch = None
        best_valid_accuracy = None
        best_test_accuracy = None
        test_accuracy_cheat = -1.0
        model_dir = None
        done = False

        for i, line in enumerate(file):
            if i == 0:
                line = line.split()
                key = ' '.join([line[21], line[9], line[20], line[7], line[12], line[13], line[6], line[24]])
                hyper_params = ' '.join(line[4:])
                continue
            if "Saved the new best model" in line:
                line = line.split()[3:]
                model_path = line[6]
                save_flag = True
                continue
            if "Epoch" in line and "start" not in line:
                line = line.split()[3:]
                epoch = line[1]
                valid_accuracy = float(line[9])
                continue
            if "Accuracy" in line:
                line = line.split()[3:]
                test_accuracy = float(line[1])

                if test_accuracy > test_accuracy_cheat:
                    test_accuracy_cheat = test_accuracy
                if save_flag:
                    best_epoch = epoch
                    best_valid_accuracy = valid_accuracy
                    best_test_accuracy = test_accuracy
                save_flag = False
            if "done" in line:
                done = True

    results[key].append({"hyper_params": hyper_params,
                         "best_epoch": best_epoch,
                         "best_valid_accuracy": best_valid_accuracy,
                         "best_test_accuracy": best_test_accuracy,
                         "test_accuracy_cheat": test_accuracy_cheat,
                         "file": log_file_path,
                         "model_path": model_path,
                         "done": done})


sorted_results = sorted(results.items(), key=lambda kv: -np.mean([e["best_valid_accuracy"] for e in kv[1]]) )
# sorted_results = sorted(results.items(), key=lambda kv: -np.mean([e["best_test_accuracy"] for e in kv[1]]) )
# sorted_results = sorted(results.items(), key=lambda kv: -np.mean([e["test_accuracy_cheat"] for e in kv[1]]) )

for k, v in sorted_results:
    print(k)
    seq = []
    for e in v:
        print(e["hyper_params"])
        print(e["best_valid_accuracy"], e["best_test_accuracy"], e["test_accuracy_cheat"], e["best_epoch"], e["file"], e["done"])
        print(e["model_path"])
        seq.append(e["best_test_accuracy"])
        # seq.append(e["test_accuracy_cheat"])
        print('---------------')

    print(seq)
    if seq:
        print(max(seq) * 100, min(seq) * 100, np.mean(seq)* 100, np.std(seq)* 100)
    print(4 * "==================")
