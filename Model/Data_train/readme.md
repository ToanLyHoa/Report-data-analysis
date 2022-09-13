*******************************************************************
*******************************************************************
- data.py: chứa hàm thực thi xử lí dữ liệu thô thành dataframe có ý nghĩa và lưu lại vào folder Saved

- epoch.py: chứa các hàm thực thi từ dataframe thành dữ liệu dạng mảng và chia ra data và nhãn để 
          dễ dàng train model

- ulti.py: chứa các hàm để xử lí dataframe

*******************************************************************
- Để có thể run trên dữ liệu thô mà không đọc file có sẵn:
          * data.py: Đổi tên đường dẫn các file trên đầu data.py
          * epoch.py: Comment lại những code đọc file có sẵn từ folder, và uncomment đoạn code từ data.prepare_data().