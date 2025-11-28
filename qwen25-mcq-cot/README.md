# Qwen 2.5 MCQ Chain-of-Thought (CoT)

## 🎯 Mục Tiêu Dự Án
Xây dựng mô hình ngôn ngữ lớn (LLM) chuyên giải quyết các câu hỏi trắc nghiệm (MCQ) đòi hỏi suy luận phức tạp (Commonsense QA).
- **Base Model**: Qwen 2.5 1.5B Instruct
- **Dataset**: ECQA (Explanations for CommonsenseQA)
- **Phương pháp**:
  1. **SFT (Supervised Fine-Tuning)**: Dạy mô hình suy luận từng bước (Chain-of-Thought).
  2. **DPO (Direct Preference Optimization)**: Tối ưu hóa mô hình bằng cách học từ các mẫu suy luận đúng (chosen) và sai (rejected).

## 🚀 Hướng Dẫn Chạy (Google Colab)

### Bước 1: Upload lên Google Drive
1. Tải toàn bộ folder dự án này về máy.
2. Upload folder lên Google Drive của bạn.
   - Ví dụ đường dẫn: `My Drive/qwen25-mcq-cot`

### Bước 2: Chạy trên Colab
1. Mở file `qwen25_SFT_DPO_Training.ipynb` bằng Google Colab.
2. Chọn **Runtime > Change runtime type > T4 GPU**.
3. Chạy lần lượt các Cell từ trên xuống dưới.

### 📋 Quy Trình (Pipeline)
Notebook sẽ tự động thực hiện các bước sau:
1. **Setup**: Cài đặt thư viện và mount Google Drive.
2. **Data Prep**: Tải và xử lý dữ liệu ECQA.
3. **Train SFT**: Fine-tune mô hình cơ bản.
4. **Generate Rejected**: Tạo mẫu sai từ mô hình SFT để phục vụ DPO.
5. **Build DPO Data**: Tạo cặp dữ liệu preference (đúng/sai).
6. **Train DPO**: Tối ưu hóa mô hình với DPO.
7. **Evaluate**: Đánh giá và so sánh kết quả (Base vs SFT vs DPO).



