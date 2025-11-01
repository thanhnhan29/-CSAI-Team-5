# Project 1 - Team 5

## Mô tả Project

Project này bao gồm các thuật toán tối ưu hóa AI:

- **ACO (Ant Colony Optimization)**: Thuật toán tối ưu bầy đàn kiến
- **Simulated Annealing**: Thuật toán luyện kim mô phỏng
- **Knapsack Problem (KP)**: Bài toán cái túi
- **Rosenbrock Function (RF)**: Hàm tối ưu Rosenbrock

## Yêu cầu hệ thống

- Python >= 3.7
- pip (Python package manager)

## Hướng dẫn Setup Project

### Bước 1: Clone repository (nếu chưa có)

```bash
git clone <repository-url>
cd CSAI
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt virtual environment
# Trên macOS/Linux:
source venv/bin/activate

# Trên Windows:
# venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies từ requirements.txt

```bash
pip install -r requirements.txt
```

File `requirements.txt` bao gồm các thư viện:

- numpy: Tính toán số học
- matplotlib: Vẽ đồ thị
- seaborn: Visualization nâng cao
- tqdm: Progress bar

### Bước 4: Cài đặt package dưới dạng editable mode

```bash
pip install -e .
```

Lệnh này sẽ:

- Đọc file `pyproject.toml`
- Cài đặt package `csai` ở chế độ development
- Cho phép bạn import các module từ bất kỳ đâu trong project

### Bước 5: Chạy chương trình

```bash
# Di chuyển vào thư mục src
cd src

# Chạy file main
python main.py
```

## Cấu trúc Project

```
CSAI/
├── README.md                 # File hướng dẫn này
├── requirements.txt          # Danh sách thư viện cần thiết
├── pyproject.toml           # Cấu hình build package
├── doc/                     # Tài liệu
├── src/                     # Source code chính
│   ├── main.py             # File chạy chính
│   ├── aco/                # Ant Colony Optimization
│   │   ├── __init__.py
│   │   ├── ant.py
│   │   └── colony.py
│   ├── simulated_annealing/ # Simulated Annealing
│   │   ├── _init__.py
│   │   └── simanneal.py
│   ├── Problem/            # Các bài toán tối ưu
│   │   ├── KP/            # Knapsack Problem
│   │   └── RF/            # Rosenbrock Function
│   └── Test/              # Test files
└── venv/                   # Virtual environment (sau khi tạo)
```

## Chạy các module riêng lẻ

### Test Rosenbrock Function

```bash
cd src/Test
python RF_test.py
```

### Chạy ACO cho Knapsack Problem

```bash
cd src
python main.py
```

## Troubleshooting

### Lỗi: Module not found

Nếu gặp lỗi `ModuleNotFoundError`, hãy đảm bảo:

1. Đã cài đặt tất cả dependencies: `pip install -r requirements.txt`
2. Đã cài package ở editable mode: `pip install -e .`
3. Virtual environment đã được kích hoạt

### Lỗi: Permission denied

Trên macOS/Linux, có thể cần dùng `python3` thay vì `python`:

```bash
python3 -m pip install -r requirements.txt
```

## Ghi chú

- Luôn kích hoạt virtual environment trước khi làm việc với project
- Sau khi thay đổi code, không cần cài lại vì đã dùng editable mode (`-e`)
- Để thoát virtual environment: `deactivate`

## Team Members

Team 5 - CSAI Project

## License

[Thêm license nếu có]
