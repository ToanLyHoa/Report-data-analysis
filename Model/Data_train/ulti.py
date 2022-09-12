import pandas as pd
import numpy as np

def rename_columns(dataframe, dataframe_name):
    """
    rename columns with patern: dataframe_name.columns_name

    Input:
        dataframe: Dataframe Pandas
        dataframe_name: string
    """
    columns = dataframe.columns
    # rename columns
    new_names = []
    for index in range(len(columns)):
        new_name = f'{dataframe_name}.{columns[index]}'
        new_names.append(new_name)
    
    dataframe.columns = new_names

def string_to_datetime(dataframe, column_name):
          """
          Change string columns to datetime columns

          Input:
              dataframe: Dataframe Pandas
              column_name: name of the columns to change type
          """
          dataframe[f'{column_name}'] \
                    = pd.to_datetime(dataframe[f'{column_name}'],
                                        format = '%Y/%m/%d %H:%M:%S')

def create_order_item_df(fhs_sales_flat_order, fhs_sales_flat_order_item, product_dim, year = 2020):
          """
          Tạo ra fhs_sales_flat_order_item_state theo năm đưa vào, 
          lấy những thông tin cần thiết và chỉ xử lí với những đơn complete

          Input:   
                    fhs_sales_flat_order: Dataframe
                    fhs_sales_flat_order_item: Dataframe
                    product_dim: Dataframe
                    year: int
          Return:
                    fhs_sales_flat_order_item_state_2020: Dataframe
          """
          # merge để lấy thông tin về sku
          fhs_sales_flat_order_item_state  \
                    = pd.merge(fhs_sales_flat_order, fhs_sales_flat_order_item,
                              left_on = 'fhs_sales_flat_order.entity_id', 
                              right_on = 'fhs_sales_flat_order_item.order_id')

          # merge để lấy thông tin về category
          fhs_sales_flat_order_item_state = pd.merge(fhs_sales_flat_order_item_state, product_dim,
                                                  left_on='fhs_sales_flat_order_item.sku', 
                                                  right_on='product_dim.sku')


          # Giả sử xét trong năm 2020
          min_year = pd.to_datetime(f'{year}/1/1',
                                        format = '%Y/%m/%d %H:%M:%S')
          max_year = pd.to_datetime(f'{year}/12/31 23:59:59',
                                        format = '%Y/%m/%d %H:%M:%S')

          condition2 = ((fhs_sales_flat_order_item_state['fhs_sales_flat_order_item.created_at'] >=  min_year)
                    &           (fhs_sales_flat_order_item_state['fhs_sales_flat_order_item.created_at'] <=  max_year))     
          fhs_sales_flat_order_item_state_2020 = fhs_sales_flat_order_item_state.loc[condition2] 

          # Xét các đơn hàng thành công
          fhs_sales_flat_order_item_state_2020 = \
                    fhs_sales_flat_order_item_state_2020\
                              .loc[(fhs_sales_flat_order_item_state_2020['fhs_sales_flat_order.state'].isin(['complete']))
                              &    (fhs_sales_flat_order_item_state_2020['fhs_sales_flat_order.status'].isin(['complete']))]
          
          # Lọc ra các cột không thiết
          fhs_sales_flat_order_item_state_2020 \
                    = fhs_sales_flat_order_item_state_2020\
                              .loc[:, ['fhs_sales_flat_order_item.created_at',
                                        'product_dim.sku', 'product_dim.cat']]

          return fhs_sales_flat_order_item_state_2020

def preprocess_order_item(fhs_sales_flat_order_item_state, threshold_amount = 300, threshold_month = 3):
          """
          Tiền xử lí với fhs_sales_flat_order_item_state, loại các sản phẩm bán ít hơn 
          threshold_amount sản phẩm, loại các sản phẩm có thời gian bán ít hơn threshold_month tháng,
          cuối cùng ta sẽ đếm xem số lượng sản phẩm bán trong ngày

          Input:
                fhs_sales_flat_order_item_state: Dataframe
                threshold_amount: int
                threshold_month: int
          Return: 
                fhs_sales_flat_order_item_state: Dataframe
          """

          # Lọc các sản phẩm < threshold_amount đơn trong năm
          # Đếm số lượng sản phẩm bán theo sku
          a = fhs_sales_flat_order_item_state.groupby('product_dim.sku').count()
          a = a.reset_index()
          a = a.loc[a['fhs_sales_flat_order_item.created_at'] > threshold_amount]
          i1 = a.set_index('product_dim.sku').index
          i2 = fhs_sales_flat_order_item_state.set_index('product_dim.sku').index
          # Lọc sản phẩm
          fhs_sales_flat_order_item_state \
                    = fhs_sales_flat_order_item_state.loc[i2.isin(i1)]

          # Tiếp tục lọc những sản phẩm có thời gian bán ngắn

          # Lấy ngày bán sớm nhất và trễ nhất
          b = fhs_sales_flat_order_item_state\
                    .groupby('product_dim.sku')['fhs_sales_flat_order_item.created_at'].agg(['min', 'max'])
          b = b.reset_index()
          string_to_datetime(b, 'min')
          string_to_datetime(b, 'max')
          days = f'{threshold_month*30} days'
          b=b.loc[(b['max'] - b['min']) > pd.Timedelta(days)]
          i1 = b.set_index('product_dim.sku').index
          i2 = fhs_sales_flat_order_item_state.set_index('product_dim.sku').index
          # Lọc sản phẩm
          fhs_sales_flat_order_item_state \
                    = fhs_sales_flat_order_item_state.loc[i2.isin(i1)]

          # Tạo biến count để có thể tính tổng
          fhs_sales_flat_order_item_state['count'] = 1

          # Group by theo ngày và sku, cat để tính tổng sản phẩm bán trong ngày
          fhs_sales_flat_order_item_state = fhs_sales_flat_order_item_state\
                    .groupby(['product_dim.sku','product_dim.cat',
                              fhs_sales_flat_order_item_state['fhs_sales_flat_order_item.created_at'].dt.date])['count'].count()
          fhs_sales_flat_order_item_state = fhs_sales_flat_order_item_state.reset_index()
          string_to_datetime(fhs_sales_flat_order_item_state, 'fhs_sales_flat_order_item.created_at')
          


          return fhs_sales_flat_order_item_state

def get_sku_list(fhs_sales_flat_order_item_state):
          """
          Lấy ra list các sku có trong fhs_sales_flat_order_item_state

          Input:
                fhs_sales_flat_order_item_state: Dataframe
          Return:
                sku_list: list
          """

          sku_list = list(set(fhs_sales_flat_order_item_state.groupby('product_dim.sku').count().index))

          return sku_list

def preprocess_product_dim(product_dim):
          """
          Handle duplicate categories in the same sku, we concat each different names in one category
          after that we concat all categories in one string.

          Xử lí duplicate category trong cùng 1 sku, chúng ta nối tất cả tên lại với nhau sau khi concat 
          category vào 1 chuỗi cách nhau bởi dấu ,

          Input:
                product_dim: Dataframe
          Return: 
                product_dim: Dataframe
          """

          # ---------------------------------------------- #
          # Tiền xử lí thông tin sản phẩm: concat các category lại với nhau

          # Đầu tiên drop hết tất cả các sản phẩm trùng 5 thuộc tính này.
          product_dim.drop_duplicates(subset = ['product_dim.sku',
                    'product_dim.cat1','product_dim.cat2',
                    'product_dim.cat3', 'product_dim.cat4',
                    'product_dim.cat5'], inplace = True)

          # fill những thuộc tính na bằng ký tự trống
          product_dim = product_dim.fillna('')

          # Groupby các thuộc tính trùng vào set để loại bỏ trùng lặp
          product_dim = product_dim.groupby('product_dim.sku')[['product_dim.cat1','product_dim.cat2',
                    'product_dim.cat3', 'product_dim.cat4',
                    'product_dim.cat5']].agg(set)
          product_dim = product_dim.reset_index()

          # Đưa set về lại string và nối với nhau bằng dấu phẩy
          product_dim['product_dim.cat1'] = product_dim['product_dim.cat1'].str.join(',')
          product_dim['product_dim.cat2'] = product_dim['product_dim.cat2'].str.join(',')
          product_dim['product_dim.cat3'] = product_dim['product_dim.cat3'].str.join(',')
          product_dim['product_dim.cat4'] = product_dim['product_dim.cat4'].str.join(',')
          product_dim['product_dim.cat5'] = product_dim['product_dim.cat5'].str.join(',')

          # Xử lí tất cả các dấu phẩy thừa: lstrip dấu phẩy phía trước, rstrip dấu phẩy phía sau
          product_dim['product_dim.cat1'] = product_dim['product_dim.cat1'].str.lstrip(',')
          product_dim['product_dim.cat2'] = product_dim['product_dim.cat2'].str.lstrip(',')
          product_dim['product_dim.cat3'] = product_dim['product_dim.cat3'].str.lstrip(',')
          product_dim['product_dim.cat4'] = product_dim['product_dim.cat4'].str.lstrip(',')
          product_dim['product_dim.cat5'] = product_dim['product_dim.cat5'].str.lstrip(',')

          product_dim['product_dim.cat1'] = product_dim['product_dim.cat1'].str.rstrip(',')
          product_dim['product_dim.cat2'] = product_dim['product_dim.cat2'].str.rstrip(',')
          product_dim['product_dim.cat3'] = product_dim['product_dim.cat3'].str.rstrip(',')
          product_dim['product_dim.cat4'] = product_dim['product_dim.cat4'].str.rstrip(',')
          product_dim['product_dim.cat5'] = product_dim['product_dim.cat5'].str.rstrip(',')

          # Nối các category lại với nhau 
          product_dim['product_dim.cat'] = product_dim['product_dim.cat1'] + ',' + product_dim['product_dim.cat2'] + ',' \
                    + product_dim['product_dim.cat3'] + ',' + product_dim['product_dim.cat4'] + ','\
                              +product_dim['product_dim.cat5']
          product_dim = product_dim.reset_index()

          # Lấy thông tin cần thiết cho product_dim
          product_dim = product_dim.loc[:, ['product_dim.sku', 'product_dim.cat']]

          # Tiếp tục xóa các dấu phẩy thừa ở phía sau
          product_dim['product_dim.cat'] = product_dim['product_dim.cat'].str.rstrip(',')
          return product_dim

def create_data_full_year(fhs_sales_flat_order_item_state, sku_list, index):
        """
        Tạo ra data cho toàn bộ năm trên 1 sku từ index trong sku_list
        
        Input: 
            fhs_sales_flat_order_item_state: Dataframe
            sku_list: list []
            index: int
        Return:
            result: Dataframe
        """

        sku = sku_list[index]

        temp = fhs_sales_flat_order_item_state.loc[fhs_sales_flat_order_item_state['product_dim.sku'] == sku]

        category = temp['product_dim.cat'].iat[0]

        datelist = pd.date_range(start='01-01-2020', end='12-31-2020', freq='1d')
        datelist = pd.DataFrame({'date': datelist})

        result = pd.merge(temp, datelist, 
                left_on='fhs_sales_flat_order_item.created_at',
                right_on='date', how = 'right')
        result['product_dim.sku'] = result['product_dim.sku'].fillna(sku)
        result['product_dim.cat'] = result['product_dim.cat'].fillna(category)
        result['count'] = result['count'].fillna(0)
        result = result.loc[:, ['product_dim.sku', 'product_dim.cat', 'count', 'date']]
        result = result.sort_values('date')

        return result
          
def get_epoch_data_15_days(data_full_year, model, day_train = 15, day_predict = 7):
    """
    Chọn tất cả các ngày trong năm hay có thể gọi là epoch cho từng năm, 
    sau đó trả ra dữ liệu 15 ngày trước đó và dữ liệu 7 ngày sau là một array 2 chiều

    Input:
        data_full_year: Dataframe
        model: model sentence to vec vietnamese-sbert
    Return:
        data_array: numpy array shape (344 or 345, 792). Trong đó 344 là số ngày trong năm, 
            792 là số chiều 768 + 1 + 1 + 15 + 7 = 785 + 7. Note: 785 là dự liệu train, 7 là nhãn

    """

    # Lấy từng khoảng gồm 22 ngày, với 15 ngày train 7 ngày dự đoán
    data_list = []
    offset = day_train + day_predict
    for i in range(len(data_full_year) - offset + 1):
            data_list.append(data_full_year[i:offset + i])

    # Chuyển dataframe thành array và đưa vào list
    data_array = []
    for data in data_list:
            array = format_data(data[:day_train], data[day_train:], model)
            data_array.append(array)

    # Sau đó đưa về array 2D
    data_array = np.asarray(data_array)
    return data_array
    pass

def random_data_15_days(data_full_year, year):
    """
    Ngẫu nhiên chọn 1 ngày trong cả năm, sau đó trả ra dữ liệu 15 ngày trước đó và dữ liệu 7 ngày sau

    Input:
        data_full_year: Dataframe
        year: int
    Return:
        data_15_days: Dataframe
        data_7_days: Dataframe

    """
    min_year = pd.to_datetime(f'{year}/1/1',
                                format = '%Y/%m/%d %H:%M:%S')
    max_year = pd.to_datetime(f'{year}/12/31 23:59:59',
                                format = '%Y/%m/%d %H:%M:%S')

    offset_start = pd.Timedelta('15 days')
    offset_end = pd.Timedelta('6 days')
    # Khởi tạo khoảng an toàn để random
    range_safe = pd.date_range(start = min_year + offset_start, end = max_year - offset_end, freq = 'D')
    index = round(np.random.rand()*len(range_safe))
    # Lấy ra 22 ngày
    range_data = pd.date_range(start = range_safe[index] - offset_start, end = range_safe[index] + offset_end, freq = 'D')

    date = pd.DataFrame({'date': range_data})

    df = pd.merge(data_full_year, date, left_on = 'date', right_on='date')

    return df.iloc[:15], df.iloc[15:]
    pass

def format_data(data_15_days, data_7_days, model):
    """
    Chuyển data về toàn là số để có thể train dễ dàng
    Cấu trúc data: data_15_days = (số đơn bán trong 15 ngày, ngày, tháng, 768 word embedding cho category)
    Cấu trúc label: data_7_days = (số đơn bán trong 7 ngày từ kế tiếp)

    Input:
        data_15_days: Dataframe
        data_7_days: Dataframe
        model: model sentence to vec vietnamese-sbert
    Return:
        data_15_days: numpy array (785 = 768 + 15 + 1 + 1)
        data_7_days: numpy array (7)
    """

    # Lấy max date và bỏ các số lựng bán từng ngày vào 1 list, tiến hành xử lí và lấy các cột cần thiết
    data_15_days = data_15_days.groupby(['product_dim.sku', 'product_dim.cat'])[['date', 'count']].agg([max, list]).reset_index()
    data_15_days.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in data_15_days.columns]
    data_15_days = data_15_days.loc[:, ['product_dim.cat', 'date|max','count|list']]
    data_15_days['day'] = data_15_days['date|max'].dt.day
    data_15_days['month'] = data_15_days['date|max'].dt.month
    data_15_days = data_15_days.loc[:, ['product_dim.cat','count|list','day','month']]
    data_15_days.columns = ['product_dim.cat','sale_15_days','day', 'month']

    # Sau đó chuyển về numpy array
    sale_15_days = data_15_days['sale_15_days'].apply(np.array)[0]
    day = data_15_days['day'].apply(np.array)
    month = data_15_days['month'].apply(np.array)
    vector_sentence = model.encode(data_15_days['product_dim.cat'])[0]
    # Chỉ lấy thông tin về số lượng bán
    data_7_days = data_7_days['count'].to_numpy()

    # data_15_days = np.concatenate((sale_15_days, day, month, vector_sentence), axis = 0)
    data = np.concatenate((sale_15_days, day, month, vector_sentence, data_7_days), axis = 0)


    return data
    return data_15_days, data_7_days

    pass