import os


def get_minimal_structure(root_dir):
    """
    生成目錄和文件的簡潔結構表示，只顯示層次分佈。
    """
    structure = {}

    # 遞歸構建目錄結構
    for root, dirs, files in os.walk(root_dir):
        relative_root = os.path.relpath(root, root_dir)
        if relative_root == ".":
            relative_root = ""  # 根目錄處理
        if relative_root not in structure:
            structure[relative_root] = {"dirs": [], "files": []}

        # 添加子目錄和文件
        structure[relative_root]["dirs"].extend(dirs)
        structure[relative_root]["files"].extend(files)

    # 格式化輸出
    def format_structure(structure, indent=0):
        result = []
        for directory, contents in structure.items():
            # 顯示目錄
            result.append(" " * indent + f"{directory}/")
            # 顯示子目錄
            for sub_dir in contents["dirs"]:
                result.append(" " * (indent + 4) + f"{sub_dir}/")
            # 顯示文件
            for file in contents["files"]:
                result.append(" " * (indent + 4) + file)
        return result

    return format_structure(structure)


# 使用示例
data_dir = "LibriSpeech/train-clean-100"  # 替換為您的資料集目錄
minimal_structure = get_minimal_structure(data_dir)

# 打印結果
print("Dataset Minimal Structure:")
for line in minimal_structure:
    print(line)
