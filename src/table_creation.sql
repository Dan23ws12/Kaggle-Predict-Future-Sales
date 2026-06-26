CREATE DATABASE kaggle_pred_future_sales_db;

CREATE TABLE item_categories(
    item_category_id: int PRIMARY KEY,
    item_category_name: varchar(40)

);

CREATE TABLE shops(
    shop_id: int PRIMARY KEY,
    shop_name: varchar(40)

);

CREATE TABLE items(
	item_id: int PRIMARY KEY,
    item_name: varchar(40),
    item_category_id: int

);

CREATE TABLE orig_sales_train(
    sales_id: int PRIMARY KEY, 
    sales_date: varchar(10), 
    sales_date_block_num: int,
    item_price: int,
    item_cnt_day: int,
    shop_id: int references shops(shop_id),
	item_id: int references items(item_id),
    item_category_id: int references item_categories(item_category_id)

);
-- the transformed version of orig_sales_train used to train the machine learning models
-- for answer submission
CREATE TABLE final_sales_train(
    sales_id: int PRIMARY KEY, 
    sales_date: date, 
    sales_date_block_num: int,
    item_price: int (item_price >= 0),
    item_cnt_day: int CHECK (item_cnt_day >= 0),
    shop_id: int references shops(shop_id),
	item_id: int references items(item_id),
    item_category_id: int references item_categories(item_category_id)

);

CREATE TABLE final_submission(
    id: int PRIMARY KEY,
    item_cnt_month: int
)