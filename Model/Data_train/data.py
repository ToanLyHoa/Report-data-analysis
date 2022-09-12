import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from spicy import stats
from . import ulti

# Đường dẫn của fhs_sales_flat_order
fhs_sales_flat_order_path = '/home/it/Desktop/NTMINH/Data/Data_Fahasa/fhs_sales_flat_order.csv'
# Đường dẫn của fhs_sales_flat_order_item
fhs_sales_flat_order_item_path = '/home/it/Desktop/NTMINH/Data/Data_Fahasa/fhs_sales_flat_order_item.csv'
# Đường dẫn của fhs_sales_flat_order_item
product_dim_path = '/home/it/Desktop/NTMINH/Data/Data_Fahasa/product_dim_path.csv'

def prepare_data(year = 2020):
          """
          Khởi tạo data chứa những mặt hàng ổn định và số lượng hàng hóa bán trong ngày

          Input:
                    year: int - năm cần lấy đơn hàng
          Output:
                    fhs_sales_flat_order_item_state_2020: Dataframe - chứa thông tin về đơn hàng ổn định trong năm 2022
                    về ngày tạo, thông tin về cat đã được làm sạch
          """

          # read file
          fhs_sales_flat_order = pd.read_csv(fhs_sales_flat_order_path)
          fhs_sales_flat_order_item = pd.read_csv(fhs_sales_flat_order_item_path)
          product_dim = pd.read_csv(product_dim_path)

          # Đổi tên cho các cột
          rename_columns = ulti.rename_columns
          rename_columns(fhs_sales_flat_order, 'fhs_sales_flat_order')
          rename_columns(fhs_sales_flat_order_item, 'fhs_sales_flat_order_item')
          rename_columns(product_dim, 'product_dim')

          # Chuyển đổi kiểu dữ liệu cho các cột thời gian
          string_to_datetime = ulti.string_to_datetime
          string_to_datetime(fhs_sales_flat_order, 'fhs_sales_flat_order.created_at')
          string_to_datetime(fhs_sales_flat_order, 'fhs_sales_flat_order.updated_at')
          string_to_datetime(fhs_sales_flat_order_item, 'fhs_sales_flat_order_item.created_at')

          # Preprocess cho product_dim
          preprocess_product_dim = ulti.preprocess_product_dim
          product_dim = preprocess_product_dim(product_dim)

          # Tao ra bảng fhs_sales_flat_order_item_state
          # Lấy thông tin về số ngày bán được hàng cho từng sản phẩm
          # Tuy nhiên những ngày bán 0 sản phẩm sẽ không có.
          create_order_item_df = ulti.create_order_item_df
          fhs_sales_flat_order_item_state_2020 \
                    = create_order_item_df(fhs_sales_flat_order, fhs_sales_flat_order_item, product_dim, year)

          # preprocess cho bảng fhs_sales_flat_order_item_state_2020
          preprocess_order_item = ulti.preprocess_order_item
          fhs_sales_flat_order_item_state_2020 = preprocess_order_item(fhs_sales_flat_order_item_state_2020)

          return fhs_sales_flat_order_item_state_2020