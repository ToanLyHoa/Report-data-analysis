import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from . import data 
from . import ulti 


def create_epoch_date(year):

         # Xử lí data thô
         #  fhs_sales_flat_order_item_state_2020 = data.prepare_data_date(year)
         # fhs_sales_flat_order_item_state_2020\
         #         .to_csv('/home/it/Desktop/NTMINH/Report-data-analysis/Data/data_train/data_preprocessing.csv')


         # Đọc file đã được xử lý sẵn ở hàm prepare data
         fhs_sales_flat_order_item_state_2020 = pd.read_csv(f'../../Data/Data/data_train/data_preprocessing_{year}.csv')
         ulti.string_to_datetime(fhs_sales_flat_order_item_state_2020, 'fhs_sales_flat_order_item.created_at')
         fhs_sales_flat_order_item_state_2020 = fhs_sales_flat_order_item_state_2020.sort_values('product_dim.sku')

         fhs_sales_flat_order_item_state_2019 = pd.read_csv(f'../../Data/Data/data_train/data_preprocessing_{year - 1}.csv')
         ulti.string_to_datetime(fhs_sales_flat_order_item_state_2019, 'fhs_sales_flat_order_item.created_at')
         fhs_sales_flat_order_item_state_2019 = fhs_sales_flat_order_item_state_2019.sort_values('product_dim.sku')

         # Lấy list các sku ra
         sku_list = ulti.get_sku_list(fhs_sales_flat_order_item_state_2020)

         print('log: Lấy list các sku ra')
         print('log: Số  sku là', len(sku_list))

         # Tạo ra thông tin bán hàng của các cat1
         df = ulti.create_data_full_year(fhs_sales_flat_order_item_state_2020, sku_list, 0, year = year)
         for index in range(1, len(sku_list)):
                           data_full_year  = \
                                       ulti.create_data_full_year(fhs_sales_flat_order_item_state_2020, sku_list, index, year = year)

                           df = pd.concat([df, data_full_year], axis=0)

         df['cat1'] = df['product_dim.cat'].str.split(',').str[0]

         cat_list = np.asarray(list(set(df['cat1'])))
         df_list = []

         for cat in cat_list:
            print(cat)
            hehe = pd.DataFrame(df.loc[df['cat1'] == cat].groupby('date')['count'].mean().reset_index())
            df_list.append(hehe)

         df_list_diff_cat = []

         for cat in cat_list:
                  print(cat)
                  hehe = pd.DataFrame(df.loc[df['cat1'] != cat].groupby('date').mean().reset_index())
                  df_list_diff_cat.append(hehe)


         # Load model sentence to vec
         vietnamese_sbert = SentenceTransformer('keepitreal/vietnamese-sbert')

         # list chứa data của tất cả sku
         data_full_sku = []

         # Xử lí số chiều của data và nhãn
         day_train = 30
         day_predict = 7
         word_embedding = 768
         data_dimension = 3*day_train + day_predict


         for index in range(len(sku_list)):
            data_full_year  = \
                     ulti.create_data_full_year(fhs_sales_flat_order_item_state_2020, sku_list, index, year = year)

            category = fhs_sales_flat_order_item_state_2020.loc[fhs_sales_flat_order_item_state_2020['product_dim.sku'] == sku_list[index]]['product_dim.cat'].iat[0]
            data_full_last_year =  \
                     ulti.create_data_full_year(fhs_sales_flat_order_item_state_2019, sku_list, index, year = year - 1, category = category)

            # Lấy cat1 đầu tiên của sản phẩm đó
            cat1 = data_full_year['product_dim.cat'].str.split(',').str[0][0]
            # Lấy index trong list các cat đã lưu
            cat_index = np.where(cat_list == cat1)[0][0]

            # Lần lượt lấy data của cat cùng cat1 (data số lượng bán trung bình trên mỗi sku trong 1 cat)
            data_same_cat_full_year = df_list[cat_index]
            data_diff_cat_full_year = df_list_diff_cat[cat_index]

            data_train_label = ulti.get_epoch_data_k_days(data_full_year, data_full_last_year, data_same_cat_full_year, 
                     data_diff_cat_full_year,
                     day_train = day_train, day_predict = day_predict)

            data_full_sku.append(data_train_label)

            print(f'log: sku thứ {index} hoàn thành')


         # Chuyển sang array và trả về dữ liệu và nhãn
         data_full_sku = np.concatenate(data_full_sku, axis = 0)

         # Ở đây ta set cứng 15 ngày nên phần nhãn sẽ tính từ 785 đến sau
         return data_full_sku[:, :data_dimension], data_full_sku[:, data_dimension:]

def create_epoch_hour():

          # Xử lí data thô
          # fhs_sales_flat_order_item_state_2020 = data.prepare_data_hour()
          # fhs_sales_flat_order_item_state_2020\
          #         .to_csv('/home/it/Desktop/NTMINH/Data/Data/data_train/data_preprocessing_hour.csv')

          # Đọc file đã được xử lý sẵn ở hàm prepare data
          fhs_sales_flat_order_item_state_2020 = pd.read_csv('/home/it/Desktop/NTMINH/Data/Data/data_train/data_preprocessing_hour.csv')
          ulti.string_to_datetime(fhs_sales_flat_order_item_state_2020, 'fhs_sales_flat_order_item.created_at')


          # Lấy list các sku ra
          # sku_list = ulti.get_sku_list(fhs_sales_flat_order_item_state_2020)
          sku_list = [8935086851760, 'qtflowerstore', 'qtsgarden0820', '8935244840506',
       '8935244842036', '8935244840490', '8935244839920', '8935244844115',
       '8935244842036-qt', 'll-8935244842036']

          # Load model sentence to vec
          vietnamese_sbert = SentenceTransformer('keepitreal/vietnamese-sbert')

          # list chứa data của tất cả sku
          data_full_sku = []

          # Xử lí số chiều của data và nhãn
          hour_train = 48
          hour_predict = 12
          word_embedding = 768
          data_dimension = word_embedding + 3 + hour_train


          for index in range(10):
                    data_full_year  = \
                              ulti.create_data_full_year(fhs_sales_flat_order_item_state_2020, sku_list, index, False, 2020)

                    data = ulti.get_epoch_data_k_hours(data_full_year, vietnamese_sbert, 
                              hour_train = hour_train, hour_predict = hour_predict)

                    data_full_sku.append(data)

          # Chuyển sang array và trả về dữ liệu và nhãn
          data_full_sku = np.concatenate(data_full_sku, axis = 0)

          # Ở đây ta set cứng 15 ngày nên phần nhãn sẽ tính từ 785 đến sau
          return data_full_sku[:, :data_dimension], data_full_sku[:, data_dimension:]


