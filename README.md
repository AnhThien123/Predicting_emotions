# Ứng dụng Nhận diện Cảm xúc Tiếng Việt (PhoBERT + Streamlit)

Ứng dụng web đơn giản: nhập một câu tiếng Việt, mô hình PhoBERT đã fine-tune trên
bộ dữ liệu UIT-VSMEC sẽ dự đoán cảm xúc (Vui vẻ / Buồn / Giận dữ / Sợ hãi /
Ngạc nhiên / Ghê tởm / Khác) kèm biểu đồ xác suất cho từng nhãn.

## 1. Chuẩn bị thư mục

Giải nén `phobert-vsmec-emotion-final.zip` (file bạn đã tải từ Colab) và đặt
thư mục kết quả **cùng cấp** với `app.py`, sao cho cấu trúc thư mục như sau:

```
your-project/
├── app.py
├── requirements.txt
└── phobert-vsmec-emotion-final/
    ├── config.json
    ├── model.safetensors  (hoặc pytorch_model.bin)
    ├── tokenizer_config.json
    ├── vocab.txt / bpe.codes
    └── ... (các file khác do trainer.save_model() tạo ra)
```

Nếu bạn đặt tên thư mục khác, mở `app.py` và sửa dòng:
```python
MODEL_DIR = "phobert-vsmec-emotion-final"
```

## 2. Cài đặt thư viện (VS Code — Terminal)

Khuyến khích tạo virtual environment trước:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`. Nhập câu tiếng Việt (có dấu)
vào ô nhập liệu và bấm **"Phân tích cảm xúc"**.

## Lưu ý

- Mô hình được huấn luyện trên tiếng Việt **có dấu** — nhập câu không dấu có
  thể cho kết quả kém chính xác hơn (đây cũng là một phát hiện đã ghi nhận
  trong phần thực nghiệm của tiểu luận).
- Lần chạy đầu tiên, `underthesea` có thể tải thêm một số dữ liệu cần thiết —
  cần kết nối Internet ở lần đầu tiên.
- Nếu máy không có GPU, mô hình vẫn chạy được trên CPU (chỉ chậm hơn một chút
  cho mỗi lần dự đoán, không đáng kể với PhoBERT-base).
