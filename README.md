# Kaggle-Predict-Future-Sales
This project is about analyzing the sales data  for the Predict Future Sales Kaggle competition.
Link: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview

The competition is about predicting the total sales for each shop of a Russian company named 1C Company for each product for a given month.

The dataset consists of:
- Sales data from 2013 to 2015
- Shop data
- Product data
- Item category data
- Item price data
- Test data (predicted sales for each shop of the company for each product for the next month)
The evaluation metric is the RMSE (Root Mean Squared Error).

The competition has ended but this project is just for me to enhance my data analysis and predictive modeling skills.

I will use the following tools:
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib

I will start by exploring the data and understanding the problem statement.

Then, I will preprocess the data and create features that will help in predicting the sales.

Finally, I will train different models and evaluate their performance using the RMSE metric.

You can copy the .env.example file and fill in the values to run the project
The files needed to run the project are in requirements.txt, you can download that onto your
sytem or into a python virtual environment 

Some ideas for features such as length of item name were obtained from the Kaggle discussion 
linked here: https://www.kaggle.com/code/abubakar624/first-place-solution-kaggle-predict-future-sales#Feature-engineering

After training models using data with negative values replaced with 0, got a score of 848.66093
Citation:
Alexander Guschin, Dmitry Ulyanov, inversion, Mikhail Trofimov, utility, and Μαριος Μιχαηλιδης KazAnova. Predict Future Sales. https://kaggle.com/competitions/competitive-data-science-predict-future-sales, 2018. Kaggle.
