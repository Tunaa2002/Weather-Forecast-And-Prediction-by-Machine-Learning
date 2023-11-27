import wx
import os
import time
import sys


input_max_temp = input("Vui lòng nhập nhiệt độ tối đa: ")
input_min_temp = input("Vui lòng nhập nhiệt độ tối thiểu: ")
input_meandew = input("Vui lòng nhập điểm sương trung bình: ")
input_meanhum = input("Vui lòng nhập độ ẩm trung bình: ")
input_pressure = input("Vui lòng nhập áp suất trung bình: ")
input_meancloud = input("Vui lòng nhập đám mây trung bình: ")
input_rainfall = input("Vui lòng nhập lượng mưa trung bình: ")
input_population = input("Vui lòng nhập mật độ dân số: ")
input_sunshine = input("Vui lòng nhập số giờ nắng trung bình: ")
input_wind_dir = input("Vui lòng nhập hướng gió trung bình: ")
input_wind_speed = input("Vui lòng nhập tốc độ gió trung bình: ")
input_air_quality = input("Vui lòng nhập chất lượng sức khỏe không khí trung bình: ")


if (True):
    # coding: utf-8

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as seabornInstance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn import metrics


    dataset = pd.read_csv('data.csv')

    dataset.shape

    dataset.describe()

    dataset.isnull().any()

    dataset = dataset.ffill()

    dataset.plot(x='pressure', y='mean_temp', style='o')
    plt.title('Áp suất và nhiệt độ trung bình')
    plt.xlabel('pressure')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/pressure.png")
    plt.show()
    dataset.plot(x='max_temp', y='mean_temp', style='o')
    plt.title('Nhiệt độ tối đa so với nhiệt độ trung bình')
    plt.xlabel('max_temp')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/max_temp.png")
    plt.show()
    dataset.plot(x='min_temp', y='mean_temp', style='o')
    plt.title('Nhiệt độ tối thiểu so với nhiệt độ trung bình')
    plt.xlabel('min_temp')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/min_temp.png")
    plt.show()
    dataset.plot(x='meandew', y='mean_temp', style='o')
    plt.title('Điểm sương trung bình so với nhiệt độ trung bình')
    plt.xlabel('meandew')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/meandew.png")
    plt.show()
    dataset.plot(x='meanhum', y='mean_temp', style='o')
    plt.title('Độ ẩm trung bình và nhiệt độ trung bình')
    plt.xlabel('meanhum')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/meanhum.png")
    plt.show()
    dataset.plot(x='meancloud', y='mean_temp', style='o')
    plt.title('Đám mây trung bình và nhiệt độ trung bình')
    plt.xlabel('meancloud')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/meancloud.png")
    plt.show()
    dataset.plot(x='rainfall', y='mean_temp', style='o')
    plt.title('Lượng mưa và nhiệt độ trung bình')
    plt.xlabel('rainfall')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/rainfall.png")
    plt.show()
    dataset.plot(x='population', y='mean_temp', style='o')
    plt.title('Dân số và nhiệt độ trung bình')
    plt.xlabel('Population')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/population.png")
    plt.show()
    dataset.plot(x='sunshine_hour', y='mean_temp', style='o')
    plt.title('Nắng chói chang và nhiệt độ trung bình')
    plt.xlabel('sunshine_hour')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/sunshine.png")
    plt.show()
    dataset.plot(x='wind_direction', y='mean_temp', style='o')
    plt.title('Hướng gió và nhiệt độ trung bình')
    plt.xlabel('wind_direction')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/wind_direction.png")
    plt.show()
    dataset.plot(x='wind_speed', y='mean_temp', style='o')
    plt.title('Hướng gió và nhiệt độ trung bình')
    plt.xlabel('wind_speed')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/wind_speed.png")
    plt.show()
    dataset.plot(x='air_health_quality', y='mean_temp', style='o')
    plt.title('Chất lượng sức khỏe không khí so với nhiệt độ trung bình')
    plt.xlabel('air_health_quality')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/air_quality.png")
    plt.show()

    X = dataset[['pressure', 'max_temp', 'min_temp', 'meandew', 'meanhum', 'meancloud', 'rainfall', 'population',
                 'sunshine_hour', 'wind_direction', 'wind_speed', 'air_health_quality']]
    y = dataset['mean_temp']

    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(dataset['mean_temp'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("Dự đoán hồi quy tuyến tính: ")

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    coeff_df.sort_values(by='Coefficient', ascending=False)

    pos_coeffs_df = coeff_df[(coeff_df['Coefficient'] >= 0)].sort_values(by='Coefficient', ascending=False)
    # pos_coeffs_df.sort_values(by='Estimated_Coefficients', ascending=False)
    pos_coeffs_df

    pos_coeffs_df = coeff_df[(coeff_df['Coefficient'] < 0)].sort_values(by='Coefficient', ascending=True)
    # pos_coeffs_df.sort_values(by='Estimated_Coefficients', ascending=False)
    pos_coeffs_df

    y_pred = regressor.predict(X_test)

    import seaborn as sns

    g = sns.regplot(y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Nhiệt độ trung bình dự đoán', ylabel='Nhiệt độ trung bình thực tế', title='Dự đoán mô hình')
    plt.title('Sơ đồ hồi quy cho giá trị thực tế và giá trị dự đoán')

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("statistics/linear_regression_comparison.png")
    plt.show()

    # R2 for train and test data
    R2_reg_train = regressor.score(X_train, y_train)
    R2_reg_test = regressor.score(X_test, y_test)
    print('R bình phương cho dữ liệu train là: %.3f' % (R2_reg_train))
    print('R bình phương cho dữ liệu test là: %.3f' % (R2_reg_test))

    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - regressor.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - regressor.predict(X_test)) ** 2))
    print('Sai số bình phương trung bình gốc của dữ liệu train là: %.3f' % (RMSE_reg_train))
    print('Sai số bình phương trung bình gốc của dữ liệu test là: %.3f' % (RMSE_reg_test))

    print('Lỗi tuyệt đối trung bình:', metrics.mean_absolute_error(y_test, y_pred))
    print('Lỗi bình phương trung bình:', metrics.mean_squared_error(y_test, y_pred))
    print('Lỗi bình phương gốc:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # input_pressure = 1000
    # input_max_temp = 30
    # input_min_temp = 25
    # input_meandew = 25
    # input_meanhum = 80
    estimated_temp = regressor.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum),float(input_meancloud),float(input_rainfall),int(input_population),float(input_sunshine),float(input_wind_dir),float(input_wind_speed),float(input_air_quality)]])
    print ("Nhiệt độ trung bình dự kiến là", estimated_temp)

    print(" ")
    print("Dự đoán hàng xóm gần nhất của K: ")

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)

    pred_knn = knn.predict(X_test)
    pred_knn
    y_pred = knn.predict(X_test)


    import seaborn as sns

    g = sns.regplot(y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Nhiệt độ trung bình dự đoán', ylabel='Nhiệt độ trung bình thực tế', title='Dự đoán mô hình')
    plt.title('Sơ đồ hồi quy cho giá trị thực tế và giá trị dự đoán')


    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1


    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("statistics/KNN_comparison.png")
    plt.show()


    # R2 for train and test data
    # R2 for train and test data
    R2_reg_train = knn.score(X_train, y_train)
    R2_reg_test = knn.score(X_test, y_test)
    print('R bình phương cho dữ liệu train là: %.3f' % (R2_reg_train))
    print('R bình phương cho dữ liệu test là: %.3f' % (R2_reg_test))


    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - knn.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - knn.predict(X_test)) ** 2))
    print('Sai số bình phương trung bình gốc của dữ liệu train là: %.3f' % (RMSE_reg_train))
    print('Sai số bình phương trung bình gốc của dữ liệu test là: %.3f' % (RMSE_reg_test))


    print('Lỗi tuyệt đối trung bình:', metrics.mean_absolute_error(y_test, y_pred))
    print('Lỗi bình phương trung bình:', metrics.mean_squared_error(y_test, y_pred))
    print('Lỗi bình phương gốc:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # In[27]:

    # input_pressure = 1000
    # input_max_temp = 30
    # input_min_temp = 25
    # input_meandew = 25
    # input_meanhum = 80
    estimated_temp = knn.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum),float(input_meancloud),float(input_rainfall),int(input_population),float(input_sunshine),float(input_wind_dir),float(input_wind_speed),float(input_air_quality)]])

    print ("Nhiệt độ trung bình dự kiến là", estimated_temp)

    print(" ")
    print("Dự đoán hồi quy rừng ngẫu nhiên: ")

    rf = RandomForestRegressor(random_state=5, n_estimators=20)
    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)
    pred_rf
    y_pred = rf.predict(X_test)

    import seaborn as sns

    g = sns.regplot(y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Nhiệt độ trung bình dự đoán', ylabel='Nhiệt độ trung bình thực tế', title='Dự đoán mô hình')
    plt.title('Sơ đồ hồi quy cho giá trị thực tế và giá trị dự đoán')

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("statistics/random_forest_comparison.png")
    plt.show()

    # R2 for train and test data
    # R2 for train and test data
    # R2 for train and test data
    R2_reg_train = rf.score(X_train, y_train)
    R2_reg_test = rf.score(X_test, y_test)
    print('R bình phương cho dữ liệu train là: %.3f' % (R2_reg_train))
    print('R bình phương cho dữ liệu test là: %.3f' % (R2_reg_test))

    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - rf.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - rf.predict(X_test)) ** 2))
    print('Sai số bình phương trung bình gốc của dữ liệu train là: %.3f' % (RMSE_reg_train))
    print('Sai số bình phương trung bình gốc của dữ liệu test là: %.3f' % (RMSE_reg_test))


    print('Lỗi tuyệt đối trung bình:', metrics.mean_absolute_error(y_test, y_pred))
    print('Lỗi bình phương trung bình', metrics.mean_squared_error(y_test, y_pred))
    print('Lỗi bình phương gốc:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    estimated_temp = rf.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum),float(input_meancloud),float(input_rainfall),int(input_population),float(input_sunshine),float(input_wind_dir),float(input_wind_speed),float(input_air_quality)]])

    print ("Nhiệt độ trung bình dự kiến là", estimated_temp)

