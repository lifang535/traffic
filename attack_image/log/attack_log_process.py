# def read_log_data(log_path):
#     with open(log_path, 'r') as file:
#         lines = file.readlines()

#     objects_before_nms = []

#     for line in lines:
#         parts = line.split(',')
#         for part in parts:
#             if 'objects_num_before_nms:' in part and '->' not in part:
#                 num = int(part.split(':')[1].strip())
#                 objects_before_nms.append(num)
#             elif '_objects_num_before_nms:' in part:
#                 num = int(part.split(':')[1].strip())
#                 objects_before_nms.append(num)

#     # Remove the first two elements
#     if len(objects_before_nms) > 2:
#         objects_before_nms = objects_before_nms[2:]

#     # Print the entire list
#     print(objects_before_nms)

# # Example call
# log_path = 'single_attack/single_attack_person_epochs_200.log' # Replace with your actual file path
# read_log_data(log_path)

log_path = 'stra_attack/stra_attack_person_epochs_200.log'

# 初始化列表
objects_before_nms_list = []
objects_after_nms_list = []
_objects_before_nms_list = []
_objects_after_nms_list = []

# 读取文件
with open(log_path, 'r') as file:
    for line in file:
        # 提取每行的数据
        parts = line.split(',')
        before_nms = int(parts[1].split(': ')[1])
        after_nms = int(parts[2].split(': ')[1])
        _before_nms = int(parts[5].split(': ')[1])
        _after_nms = int(parts[6].split(': ')[1])
        objects_before_nms_list.append(before_nms)
        objects_after_nms_list.append(after_nms)
        _objects_before_nms_list.append(_before_nms)
        _objects_after_nms_list.append(_after_nms)

# # 删除前两个元素
# objects_before_nms_list = objects_before_nms_list[2:]
# objects_after_nms_list = objects_after_nms_list[2:]
# _objects_before_nms_list = _objects_before_nms_list[2:]
# _objects_after_nms_list = _objects_after_nms_list[2:]

# 打印结果
print("objects_before_nms_list =", objects_before_nms_list)
print("objects_after_nms_list =", objects_after_nms_list)
print("_objects_before_nms_list =", _objects_before_nms_list)
print("_objects_after_nms_list =", _objects_after_nms_list)