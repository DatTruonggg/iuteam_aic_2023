import os
import json

root_directory = 'D:/DatTruong/All/2025/HCM_AI/Data'  # Đường dẫn đến thư mục gốc của bạn

# Hàm để tạo thông tin cho một folder cụ thể
def create_folder_info(folder_path, folder_name, current_index):
    folder_info = {}
    image_paths = []  # Danh sách các đường dẫn hình ảnh trong folder này

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # Sắp xếp các đường dẫn hình ảnh
    image_paths.sort()

    # Tạo danh sách id cho các hình ảnh dựa trên tên tệp
    list_shot_id = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]

    # Chia thành các nhóm có 6 shot_id mỗi nhóm
    chunk_size = 6
    list_shot_id_chunks = [list_shot_id[i:i + chunk_size] for i in range(0, len(list_shot_id), chunk_size)]
    list_shot_path_chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    # Tạo thông tin cho từng nhóm
    for i, (shot_id_chunk, shot_path_chunk) in enumerate(zip(list_shot_id_chunks, list_shot_path_chunks)):
        list_shot_info = [{"shot_id": shot_id, "shot_path": path.replace('\\', '/').split("Data/")[1]} for shot_id, path in zip(shot_id_chunk, shot_path_chunk)]
        folder_info[i + current_index] = {
            "image_path": list_shot_info[0]["shot_path"],
            "list_shot_id": shot_id_chunk,
            "list_shot_path": list_shot_info
        }

    return folder_info

# Tạo thông tin cho tất cả các subfolders từ Keyframes_L01 đến Keyframes_L10
data_info = {}
current_index = 0
folder_names = [f"Keyframes_L{i:02d}" for i in range(1, 37)]
folder_names = sorted(folder_names)  # Sắp xếp theo thứ tự số học
for subfolder_name in folder_names:
    subfolder_path = os.path.join(root_directory, subfolder_name)
    if os.path.isdir(subfolder_path):
        data_info[subfolder_name] = create_folder_info(subfolder_path, subfolder_name, current_index)
        current_index += len(data_info[subfolder_name])

# Lưu thông tin vào một tệp JSON
output_json_path = 'Keyframes_Info.json'
with open(output_json_path, 'w') as json_file:
    json.dump(data_info, json_file, indent=4)

print(f"Saved: {output_json_path}")
