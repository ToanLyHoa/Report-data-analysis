from data import prepare_data 
from . import ulti 
import pandas as pd
import torch        
from sentence_transformers import SentenceTransformer
import numpy as np


def create_epoch():

          # Xử lí data thô
          # fhs_sales_flat_order_item_state_2020 = prepare_data()
          # fhs_sales_flat_order_item_state_2020\
          #         .to_csv('/home/it/Desktop/NTMINH/Report-data-analysis/Data/data_train/data_preprocessing.csv')


          # Đọc file đã được xử lý sẵn ở hàm prepare data
          fhs_sales_flat_order_item_state_2020 = pd.read_csv('/home/it/Desktop/NTMINH/Report-data-analysis/Data/data_train/data_preprocessing.csv')
          ulti.string_to_datetime(fhs_sales_flat_order_item_state_2020, 'fhs_sales_flat_order_item.created_at')

          # Lấy list các sku ra
          sku_list = ulti.get_sku_list(fhs_sales_flat_order_item_state_2020)

          # Load model sentence to vec
          vietnamese_sbert = SentenceTransformer('keepitreal/vietnamese-sbert')

          # list chứa data của tất cả sku
          data_full_sku = []

          # Xử lí số chiều của data và nhãn
          day_train = 30
          day_predict = 7
          word_embedding = 768
          data = word_embedding + 2 + day_train


          for index in range(len(sku_list)):
                    data_full_year  = \
                              ulti.create_data_full_year(fhs_sales_flat_order_item_state_2020, sku_list, index)

                    data = ulti.get_epoch_data_k_days(data_full_year, vietnamese_sbert, 
                              day_train = day_train, day_predict = day_predict)

                    data_full_sku.append(data)

          # Chuyển sang array và trả về dữ liệu và nhãn
          data_full_sku = np.concatenate(data_full_sku, axis = 0)

          # Ở đây ta set cứng 15 ngày nên phần nhãn sẽ tính từ 785 đến sau
          return data_full_sku[:, :data], data_full_sku[:, data:]

