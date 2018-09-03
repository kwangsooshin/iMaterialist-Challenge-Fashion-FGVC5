import csv
import os
import ast


num_of_label = 228

root = "/workspace/dataset/FGVC5_Fashion"
dir = "test"


base_dir = os.path.join(root, dir)
base_csv = os.path.join(root, dir + ".csv")

with open(base_csv, 'wb') as f:
    wr = csv.writer(f)

    file_list = os.listdir(base_dir)

    if dir == "test":
        wr.writerow(["ID"])

        for file_ in file_list:
            file_name, file_ext = os.path.splitext(file_)
            idx = "{0:07d}".format(int(file_name))
            file_src = os.path.join(base_dir, file_)
            file_dst = os.path.join(base_dir, idx + file_ext)
            row_list = [idx]
            wr.writerow(row_list)
            os.rename(file_src, file_dst)

    else:
        wr.writerow(["ID"] + [str(x + 1) for x in range(num_of_label)])

        for file_ in file_list:
            file_name, file_ext = os.path.splitext(file_)
            file_split = file_name.split("_")
            file_id = int(file_split[1])
            file_label = ast.literal_eval(file_split[3])

            # 1,014,544 => 7
            idx = "{0:07d}".format(file_id)
            label = [str(0) for x in range(num_of_label)]
            for l in file_label:
                label[l - 1] = str(1)

            # 228
            row_list = [idx] + label
            wr.writerow(row_list)

            file_src = os.path.join(base_dir, file_)
            file_dst = os.path.join(base_dir, idx + file_ext)
            os.rename(file_src, file_dst)




