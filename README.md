# Hệ Thống Đề Xuất Sách Sử Dụng Giải Thuật KNN

## Giới Thiệu

Đề tài này tập trung vào việc xây dựng một hệ thống đề xuất sách sử dụng giải thuật K-Nearest Neighbors (KNN). Hệ thống này giúp người đọc tìm ra những cuốn sách phù hợp với sở thích và phong cách của họ dựa trên các đặc điểm đã được thu thập từ trước. KNN là một trong những giải thuật học máy không giám sát phổ biến, được sử dụng để phân loại và hồi quy. Trong ngữ cảnh hệ thống đề xuất, KNN sẽ giúp tìm ra những cuốn sách tương tự dựa trên khoảng cách giữa các đặc điểm.

## Link Dataset

Dữ liệu sử dụng cho hệ thống đề xuất sách có thể được tải từ liên kết sau:
[Dataset](https://drive.google.com/drive/folders/19Yo9kM1HLWCUejRYvXSmlBVjtervuV2Z?usp=sharing)

## Các Bước Cài Đặt

Để cài đặt và chạy hệ thống đề xuất sách sử dụng giải thuật KNN, bạn có thể làm theo các bước sau:

1. **Clone Repository**

   ```bash
   git clone https://github.com/yourusername/book-recommendation-system.git
   cd book-recommendation-system
   
2. **Tải Dataset**
   Tạo một thư mục có tên **Data** trong clone folder và đưa [Dataset](https://drive.google.com/drive/folders/19Yo9kM1HLWCUejRYvXSmlBVjtervuV2Z?usp=sharing) vào bên trong.

3. **Chạy hệ thống**
   ```bash
   python main.py
   
4. **Kết quả kiểm tra**
   Hệ thống sẽ đưa ra các đề xuất dựa trên dữ liệu đầu vào và xuất kết quả trên terminal.