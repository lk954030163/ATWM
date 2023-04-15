import random, csv, sys


def write2csv(csvdata, filename):
    random.shuffle(csvdata)
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        for row in csvdata:
            writer.writerow(row)
        print("OK！")


def csv2data(filename):
    """
    """
    data = []
    f = open(filename, "r", encoding="UTF-8")
    line = f.readline()  # 读取第一行
    while line:
        line = line.split(",")
        data.append([line[0], str(int(line[1]))])
        line = f.readline()  # 读取下一行
    f.close()
    return data


def add_csv(trains, adv_data):
    adv_data_name = adv_data
    trains = csv2data(trains)
    adv_data = csv2data(adv_data)
    data = trains + adv_data
    write2csv(data, adv_data_name)


def delete_csv(trains, adv_data):
    adv_data_name = adv_data
    trains = csv2data(trains)
    adv_data = csv2data(adv_data)
    new_adv_data = []
    for data in adv_data:
        if data not in trains:
            new_adv_data.append(data)

    write2csv(new_adv_data, adv_data_name)


"""
trains = str(sys.argv[1])
adv_data = str(sys.argv[2])
# delete_csv(trains, adv_data)
add_csv(trains, adv_data)
"""
