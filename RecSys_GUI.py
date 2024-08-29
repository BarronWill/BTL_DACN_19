import tkinter as tk
from tkinter import messagebox
from recs.CF import *
from recs.CBF import *
from recs.HBF import *
# A00540411RKGTDNU543WS
data = pd.read_csv('Data/new_book_Rating_095.csv').drop_duplicates(['User_id', 'Title'])

# Hàm chuyển sang giao diện 2
def switch_to_interface_2():
    user_id = user_id_entry.get()
    recommended_books = colab_based(data,
                                    user_id=user_id,
                                    num_recommend=5
                                    )
    based_on_preferences_books = content_based(data,
                                               user_id=user_id,
                                               num_recommend=5
                                               )
    explore_more_books = weighted_hybrid(data,
                                         recommended_books,
                                         based_on_preferences_books,
                                         n_recommend=5, colab_weight=0.6,
                                         content_weight=0.4)
    # Danh sách sách mẫu
    if user_id:
        interface_1.pack_forget()
        interface_2.pack(fill='both', expand=True)
        add_slider(interface_2, "Có thể bạn sẽ thích", recommended_books)
        add_slider(interface_2, "Dựa trên sở thích của bạn", based_on_preferences_books)
        add_slider(interface_2, "Khám phá thêm", explore_more_books)
    else:
        messagebox.showwarning("Lỗi", "Vui lòng nhập ID người dùng")

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Hệ thống đề xuất sách")
root.geometry("600x400")

# Giao diện 1
interface_1 = tk.Frame(root)
interface_1.pack(fill='both', expand=True)

user_id_label = tk.Label(interface_1, text="Nhập ID người dùng:")
user_id_label.pack(pady=10, anchor='w', padx=10)

user_id_entry = tk.Entry(interface_1)
user_id_entry.pack(pady=10, padx=10)

recommend_button = tk.Button(interface_1, text="Đề xuất sách", command=switch_to_interface_2)
recommend_button.pack(pady=20)

# Giao diện 2
interface_2 = tk.Frame(root)

# Hàm để thêm slider
def add_slider(frame, title, books):
    title_label = tk.Label(frame, text=title, anchor='w')
    title_label.pack(fill='x', padx=10, pady=(10, 10))

    # Tạo một khung để chứa canvas và scrollbar
    container = tk.Frame(frame)
    container.pack(fill='x', padx=50, pady=(0, 10))

    # Tạo canvas
    canvas = tk.Canvas(container, height=50)
    canvas.pack(side="left", fill="x", expand=True)

    # Thêm scrollbar
    scrollbar = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
    scrollbar.pack(side="bottom", fill="x")
    canvas.config(xscrollcommand=scrollbar.set)

    # Tạo frame bên trong canvas
    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor='nw')

    # Thêm tên sách vào frame bên trong canvas
    for book in books:
        book_title = book[0]
        book_label = tk.Label(inner_frame, text=book_title, padx=10, pady=0, height= 4, relief="raised", width=25, wraplength=200)
        book_label.pack(side="left", padx=5)

    # Đặt kích thước cho inner_frame
    inner_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


# Chạy ứng dụng
root.mainloop()
