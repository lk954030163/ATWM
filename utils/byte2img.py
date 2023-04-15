import math
from PIL import Image


def create_file(data, size, image_type):
    """ """
    try:
        image = Image.new(image_type, size)
        image.putdata(data)
        return image

    except Exception as err:
        print(err)


def b2img(filename):
    tables = []
    f = open(filename, "r", encoding="UTF-8")
    line = f.readline()  # 读取第一行
    while line:
        line = line.split(" ")
        for i in range(1, len(line) - 1):
            if line[i] == "??":
                continue
            num = int(line[i], 16)
            tables.append(num)  # 列表增加
        line = f.readline()  # 读取下一行
    f.close()
    size = int(math.sqrt(len(tables)))
    img = create_file(tables, (size + 1, size + 1), "L")

    return img
    # img.save(filename+".jpg")


def getBinaryData(filename):
    """
    Extract byte values from binary executable file and store them into list
    param filename: PE executable file name
    return: byte value list
    """
    binary_values = []

    with open(filename, "rb") as fileobject:

        # read file byte by byte
        data = fileobject.read(1)

        while data != b"":
            binary_values.append(ord(data))
            data = fileobject.read(1)

    return binary_values
