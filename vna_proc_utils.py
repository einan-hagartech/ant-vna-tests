import matplotlib.pyplot as plt
import csv
import numpy as np
from numpy import random
import pandas as pd
from sklearn.linear_model import Lasso, MultiTaskLasso, LassoCV, LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

db_scans = "gluk_db.vna_scans.json"
db_patients = "gluk_db.patients.json"
db_antennas = "gluk_db.add_antena.json"

def plot_res(res, fignum, show = 0):
    plt.figure(fignum)
    f = res['FREQ']
    s = res['scan_data'][-1]
    plt.plot(f,s)
    if show:
        plt.show()


def add_additional_data(filename, do_plot, title_str = []):
    reslist = []
    qual_list = []
    proto = []
    k = 0
    with open(filename, 'r', newline='') as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            y = [float(num) for num in row['MEASUREMENT'][1:-1].split()]
            y = np.array(y)
            if k == 0: # this assumes that the first sample was good and can serve as a prototype
                proto = y
            qual = np.mean((proto - y)**2)
            qual_list.append(qual)
            reslist.append(y)
            k += 1

    print(f"Found total {len(reslist)} measurements")
    t1 = 5 # percentile threshold for good measurements
    if do_plot:
        plt.figure(1)
        plt.hist(qual_list, 50)

        for k in range(len(reslist)):
            if qual_list[k] <= t1:
                plt.figure(2)
            else:
                plt.figure(3)
            #plt.plot(np.linspace(18,26,np.shape(reslist)[1]), reslist[k])
            plt.plot(reslist[k])

        plt.figure(1)
        plt.title(title_str)
        plt.figure(2)
        plt.title(title_str)
        plt.figure(3)
        plt.title(title_str)
        plt.show()
    valid_column = [(x <= t1) for x in qual_list]
    print(f"Found total {np.sum((np.array(qual_list) <= t1))} valid measurements")
    print(f"Found total {np.sum((np.array(qual_list) > t1))} invalid measurements")
    return valid_column

def resonance_predict(filename, measurement_range, valid_only=1, subjects_id=None, method = 'SVR'):
    X, y = get_data_from_file(filename, valid_only, subjects_id)
    lasso_Alpha = 0.01
    SVR_C = 100
    #plt.plot(X.T)
    #plt.show()
    #return
    w0 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    #res_index = [69, 172, 301]
    res_index = [316]
    reson_indx_mat = []
    reson_val_mat = []
    y_change_mat = []

    for i in range(X.shape[0]):
        v = X[i,:]
        y_change_list = []
        reson_indx_vect = []
        reson_val_list = []
        for ri in res_index:
            w = w0 + ri
            v_min_i = np.argmin(v[w]) - int((len(w0)-1)/2) + ri
            x_min = v_min_i - (v[v_min_i+1] - v[v_min_i-1])/(2*(v[v_min_i+1] + v[v_min_i-1] - 2*v[v_min_i]))
            v_min = v[v_min_i] - ((v[v_min_i+1] - v[v_min_i-1])**2) / (8*(v[v_min_i+1] + v[v_min_i-1] - 2*v[v_min_i]))
            reson_indx_vect.append(x_min)
            y0_interp = np.interp(x_min, np.arange(len(v)), v)
            y1_interp = np.interp(x_min + 1, np.arange(len(v)), v)
            ym1_interp = np.interp(x_min - 1, np.arange(len(v)), v)
            y_change = (y1_interp + ym1_interp)/2 - y0_interp
            y_change_list.append(y_change)
            reson_val_list.append(v_min)
        y_change_mat.append(y_change_list)
        reson_indx_mat.append(reson_indx_vect)
        reson_val_mat.append(reson_val_list)

    y_change_mat = np.array(y_change_mat)
    reson_indx_mat = np.array(reson_indx_mat)
    reson_val_mat = np.array(reson_val_mat)
    #X = np.hstack((y_change_mat, reson_indx_mat, reson_val_mat))
    X = np.hstack((y_change_mat, reson_indx_mat))
    #X = reson_indx_mat
    if method == 'Lasso':
        lasso1 = Lasso(alpha=lasso_Alpha, max_iter=10 ** 6)
        lasso1_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', lasso1)
        ])
        lasso1_pipeline.fit(X, y)
        y_pred = lasso1_pipeline.predict(X)
    else:
        model = SVR(kernel='linear', C=SVR_C, epsilon=0.1)
        model.fit(X, y)
        y_pred = model.predict(X)


    re = abs(y - y_pred) / y
    mard = np.mean(re)
    precentage_re = np.mean(re < 0.2)
    re_in_range = re[(np.array(y) >= measurement_range[0]) & (np.array(y) <= measurement_range[1])]
    mard_in_range = np.mean(re_in_range)
    precentage_re_in_range = np.mean(re_in_range < 0.2)
    print(f'{method} prediction MARD: {mard}')
    print(f'{method} prediction in ZONE A: {precentage_re}')
    print(f'{method} MARD in range {measurement_range}: {mard_in_range}')
    print(f'Prediction in ZONE A in range: {precentage_re_in_range}')

    #plt.figure()
    #plt.plot(X, y, 'o')

    plt.figure()
    plt.plot(y, y_pred, 'o')
    plt.plot([70, 200], [70 * 0.8, 200 * 0.8], color='black')
    plt.plot([70, 200], [70 * 1.2, 200 * 1.2], color='black')
    lim_low = measurement_range[0] * 0.95
    lim_high = measurement_range[1] * 1.05
    # plt.xlim(lim_low, lim_high)
    # plt.ylim(lim_low, lim_high)
    plt.xlim(50, 280)
    plt.ylim(lim_low, lim_high)
    plt.grid()
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'{method} Model Predictions')

    plt.figure()
    plt.plot(np.arange(len(y)), y, label='glucose')
    plt.plot(np.arange(len(y)), y_pred, label='predicted')
    plt.grid()
    plt.xlabel('Measurement')
    plt.ylabel('Glucose levels')
    plt.title('Single frequency Predictions')

    # Using np.abs() to make coefficients positive.
    if method == 'Lasso':
        lasso1_coef = lasso1.coef_

        # plotting the Column Names and Importance of Columns.
        # In Lasso Feature Selection, the importance is the absolute value of the coefficients.
        plt.figure()
        x_labels = range(len(lasso1_coef))
        colors = ['blue' if coef > 0 else 'red' for coef in lasso1_coef]
        plt.bar(x_labels, abs(lasso1_coef), color=colors)
        plt.xticks(rotation=90)
        plt.grid()
        plt.title("Feature Selection Based on Lasso")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        # Print the coefficients of the Lasso model as a table
        coef_df = pd.DataFrame({
            'Coefficient': lasso1_coef
        }, index=x_labels)

        print(f'Lasso Regression Model Score: {lasso1.score(X, y)}')
        print(f'Lasso Regression Model coefficients')
        print(f'-----------------------------------')
        print(coef_df)
        significant_coef_df = coef_df[coef_df['Coefficient'] != 0]
        print(f'-----------------------------------')
        print()
        print(f'Significant coefficients')
        print(f'-----------------------------------')
        print(significant_coef_df)
        print(f'-----------------------------------')
        print()

    plt.show()




def single_freq_predict(filename, measurement_range, valid_only=1, single_frequency_indx = None, subjects_id=None):
    X, y = get_data_from_file(filename, valid_only, subjects_id)
    #Xmean = np.mean(X.T, axis=1)
    #Xdev = X - Xmean[np.newaxis, :]
    mard_opt = 10000
    best_indx = 0
    if single_frequency_indx == None:
        sweep_vect = range(X.shape[1])
    else:
        sweep_vect = [single_frequency_indx]

    for freq_indx in sweep_vect:
        x = X[:, freq_indx]
        x = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        re = abs(y - y_pred) / y
        mard = np.mean(re)
        if mard < mard_opt:
            mard_opt = mard
            best_indx = freq_indx
    #best_indx = 144
    print(f"MARD at frequency index: {best_indx} is: {mard_opt}")

    x = X[:, best_indx]
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    #plt.plot(Xdev.T)
    #plt.show()
    #return
    plt.figure()
    plt.plot(y, y_pred, 'o')
    plt.plot([70, 200], [70 * 0.8, 200 * 0.8], color='black')
    plt.plot([70, 200], [70 * 1.2, 200 * 1.2], color='black')
    lim_low = measurement_range[0] * 0.95
    lim_high = measurement_range[1] * 1.05
    plt.xlim(lim_low, lim_high)
    plt.ylim(lim_low, lim_high)
    plt.grid()
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Single frequency Predictions')

    plt.figure()
    plt.plot(np.arange(len(y)), y, label = 'glucose')
    plt.plot(np.arange(len(y)), y_pred, label = 'predicted')
    plt.grid()
    plt.xlabel('Measurement')
    plt.ylabel('Glucose levels')
    plt.title('Single frequency Predictions')
    plt.show()


def get_data_from_file(filename, valid_only, subjects_id):
    df = pd.read_csv(filename)
    if valid_only == 1:  # remove bad measurement
        df = df[df['VALID'] == 1]
    if subjects_id:
        df = df[df['SUBJECT_ID'].isin(subjects_id)]  # accept all ids which are in subjecs_id
    y = df['GLUCOSE']
    x = df['MEASUREMENT']
    X = np.zeros([x.shape[0], len([p.strip for p in x.values[0][1:-1].split()])])
    k = 0
    for s in x:
        s = s[1:-1]
        str_list = s.split()
        str_list = [x.strip() for x in str_list]
        for n in range(len(str_list)):
            X[k, n] = float(str_list[n])
        k += 1
    return X, y

def feature_extract(filename, measurement_range, valid_only=1, lasso_Alpha = None, subjects_id=None): # updn_only can be None, 'UP' or 'DN', set subjects_id to the list of included subjects or empty for all
    # Use the Lasso model from sklearn to perform feature selection
    # Use the Lasso regression model with Cross-Validation (CV) and 5grid-search to find the best alpha parameter:
    target_n_coeffs = 10
    if lasso_Alpha == None:
        alphas = np.logspace(-0.5, 1.5, 50) # select alpha to achieve target_n_coeffs coefficients
    else:
        alphas = [lasso_Alpha] # force alpha

    X, y = get_data_from_file(filename, valid_only, subjects_id)
    for alpha in alphas:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Step 1: scale features
            ('lasso', Lasso(alpha=alpha, max_iter=10000))  # Step 2: LassoCV with 5-fold CV
        ])
        pipeline.fit(X, y)
        coefs = pipeline.named_steps['lasso'].coef_
        n_coefs = len(coefs[coefs != 0])
        if n_coefs <= target_n_coeffs:
            break
    nonzero_coeffs_ind = np.nonzero(coefs)
    # After selecting the features we run Lasso again, but per-subject with the non-zero coefficients only
    # This ensures that all models uses exactly the same frequencies
    # To discourage the new model from further reducing coefficients alpha is scaled down
    for sid in subjects_id:
        X, y = get_data_from_file(filename, valid_only, [sid])
        Xs = X[:, nonzero_coeffs_ind[0]]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Step 1: scale features
            ('lasso', Lasso(alpha=alpha/5, max_iter=10000))  # Step 2: LassoCV with 5-fold CV
        ])
        pipeline.fit(Xs, y)
        coefs = pipeline.named_steps['lasso'].coef_
        if len(coefs[coefs != 0]) != n_coefs:
            print(f"Warning: model has less coefficients than targeted for subject {sid}")
        y_pred = pipeline.predict(Xs)

        # using Mean Square Error to evaluate the model
        re = abs(y - y_pred) / y
        mard = np.mean(re)
        precentage_re = np.mean(re < 0.2)
        re_in_range = re[(np.array(y) >= measurement_range[0]) & (np.array(y) <= measurement_range[1])]
        mard_in_range = np.mean(re_in_range)
        precentage_re_in_range = np.mean(re_in_range < 0.2)
        print(f"Performance for subject {sid}")
        print(f"Total {len(y)} measurements")
        print(f'Lasso prediction MARD: {mard}')
        print(f'Lasso prediction in ZONE A: {precentage_re}')
        print(f'Prediction MARD in range {measurement_range}: {mard_in_range}')
        print(f'Prediction in ZONE A in range: {precentage_re_in_range}')
        plt.figure()
        plt.plot(y, y_pred, 'o')
        plt.plot([70, 200], [70*0.8, 200*0.8], color = 'black')
        plt.plot([70, 200], [70 * 1.2, 200 * 1.2], color = 'black')
        lim_low = measurement_range[0] * 0.95
        lim_high = measurement_range[1] * 1.05
        plt.xlim(50, 280)
        plt.ylim(lim_low, lim_high)
        plt.grid()
        plt.xlabel('True [mg/dL]')
        plt.ylabel('Predicted [mg/dL]')
        plt.title(f'LASSO Model Predictions for subject {sid}')
        plt.savefig(f".\\figures\\subject{sid}.jpg")


        # Using np.abs() to make coefficients positive.
        lasso1_coef = pipeline.named_steps['lasso'].coef_

        # plotting the Column Names and Importance of Columns.
        # In Lasso Feature Selection, the importance is the absolute value of the coefficients.
        plt.figure()
        x_labels = range(len(lasso1_coef))
        colors = ['blue' if coef > 0 else 'red' for coef in lasso1_coef]
        plt.bar(x_labels, abs(lasso1_coef), color=colors)
        plt.xticks(rotation=90)
        plt.grid()
        plt.title("Feature Selection Based on Lasso")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        # Print the coefficients of the Lasso model as a table
        print(f'Used frequency indices: {nonzero_coeffs_ind[0]}')
        print(f'-----------------------------------')
        print()

        plt.show()


def random_prediction(num_samples, measurement_range):
    y = random.randint(measurement_range[1]-measurement_range[0], size=(num_samples)) + measurement_range[0]
    y_pred = random.randint(measurement_range[1]-measurement_range[0], size=(num_samples)) + measurement_range[0]
    dumb_pred = np.ones(num_samples)*(measurement_range[0]+measurement_range[1])/2
    re = abs(y - y_pred) / y
    mard = np.mean(re)
    precentage_re = np.mean(re < 0.2)
    re_dumb = abs(y - dumb_pred) / y
    mard_dumb = np.mean(re_dumb)
    precentage_re_dumb = np.mean(re_dumb < 0.2)
    print('Random prediction')
    print(f'MARD: {mard}')
    print(f'In ZONE A: {precentage_re}')
    print(f'Trivial MARD: {mard_dumb}')
    print(f'Trivial in ZONE A: {precentage_re_dumb}')

# type can be 'lasso' or 'single_freq'
def lasso_performance_on_noise(num_samples, num_features, measurement_range, type = 'lasso', target_num_freq = 10):
    y = random.randint(measurement_range[1]-measurement_range[0], size=(num_samples)) + measurement_range[0]
    X = np.zeros((len(y), num_features))
    stdev = 1
    mean = 0
    for k in range(len(y)):
        X[k,:] = np.random.normal(loc=mean, scale=stdev, size=num_features)

    if type == 'single_freq':
        mard_opt = 1
        for freq_indx in range(len(y)):
            x = X[:, freq_indx]
            x = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            re = abs(y - y_pred) / y
            mard = np.mean(re)
            if mard < mard_opt:
                mard_opt = mard
                best_indx = freq_indx
            # best_indx = 144
        precentage_re = np.mean(re < 0.2)
        print(f"MARD at frequency index: {best_indx} is: {mard_opt}")
        print(f'Prediction in ZONE A: {precentage_re}')

    elif type == 'lasso':
        target_num_freq = 10
        alphas = np.logspace(-1, 2, 100)  # select alpha to achieve target_num_freq coefficients
        for alpha in alphas:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Step 1: scale features
                ('lasso', Lasso(alpha=alpha, max_iter=10000))  # Step 2: LassoCV with 5-fold CV
            ])
            pipeline.fit(X, y)
            coefs = pipeline.named_steps['lasso'].coef_
            n_coefs = len(coefs[coefs != 0])
            if n_coefs <= target_num_freq:
                break
        nonzero_coeffs_ind = np.nonzero(coefs)
        Xs = X[:, nonzero_coeffs_ind[0]]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Step 1: scale features
            ('lasso', Lasso(alpha=alpha / 5, max_iter=10000))  # Step 2: LassoCV with 5-fold CV
        ])
        pipeline.fit(Xs, y)
        coefs = pipeline.named_steps['lasso'].coef_
        if len(coefs[coefs != 0]) != n_coefs:
            print(f"Warning: model has less coefficients than targeted")
        y_pred = pipeline.predict(Xs)

        # using Mean Square Error to evaluate the model
        re = abs(y - y_pred) / y
        mard = np.mean(re)
        precentage_re = np.mean(re < 0.2)
        print(f'Lasso prediction MARD: {mard}')
        print(f'Lasso prediction in ZONE A: {precentage_re}')
        print(f'Used frequency indices: {list(nonzero_coeffs_ind[0])}')
        print(f'-----------------------------------')
        print()
    else:
        print(f" Error unknown type {type}")

    plt.figure()
    plt.plot(y, y_pred, 'o')
    plt.plot([70, 200], [70 * 0.8, 200 * 0.8], color='black')
    plt.plot([70, 200], [70 * 1.2, 200 * 1.2], color='black')
    lim_low = measurement_range[0] * 0.95
    lim_high = measurement_range[1] * 1.05
    plt.xlim(lim_low, lim_high)
    plt.ylim(lim_low, lim_high)
    plt.grid()
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Single frequency Predictions')
    plt.show()

def geris_library_prediction(filename, measurement_range, freq, valid_only=1):
    df = pd.read_csv(filename)
    if valid_only == 1:  # remove bad measurement
        df = df[df['VALID'] == 1]
    y = df['GLUCOSE']
    X = df['MEASUREMENT']
    f = np.linspace(18000,26000, len([p.strip for p in X.values[0][1:-1].split()]))
    x = []
    for s in X:
        s = s[1:-1]
        str_list = s.split()
        str_list = [float(x.strip()) for x in str_list]
        measure = np.interp(freq, f, np.array(str_list))
        x.append(measure)
    x = pd.Series(x, index = y.index)

    y_low = y[(y <= (measurement_range[0]+3)) & (y >= (measurement_range[0]-3))]
    x_low = x[(y <= (measurement_range[0]+3)) & (y >= (measurement_range[0]-3))]
    y_high = y[(y <= (measurement_range[1]+3)) & (y >= (measurement_range[1]-3))]
    x_high = x[(y <= (measurement_range[1]+3)) & (y >= (measurement_range[1]-3))]


    plt.figure()
    plt.scatter(y_low, x_low, color = 'b')
    plt.scatter(y_high, x_high, color='r')
    plt.grid()
    plt.xlabel('Reference measurement')
    plt.ylabel('S11 [dB]')
    plt.title('S11 at measurement frequency')
    plt.show()
    inp = input("Enter reference dB point for low glucose:")
    p_low = float(inp)
    inp = input("Enter reference dB point for low glucose:")
    p_high = float(inp)
    slope = (measurement_range[1] - measurement_range[0])/(p_high - p_low)
    intercept = measurement_range[0] - slope * p_low
    y_pred = x * slope + intercept

    plt.figure()
    plt.scatter(y[(y <= measurement_range[1]) & (y >= measurement_range[0])], y_pred[(y <= measurement_range[1]) & (y >= measurement_range[0])])
    plt.plot(y[(y <= measurement_range[1]) & (y >= measurement_range[0])], y[(y <= measurement_range[1]) & (y >= measurement_range[0])]*1.2)
    plt.plot(y[(y <= measurement_range[1]) & (y >= measurement_range[0])], y[(y <= measurement_range[1]) & (y >= measurement_range[0])]*0.8)
    lim_low = measurement_range[0]*0.8
    lim_high = measurement_range[1] * 1.2
    plt.xlim(lim_low, lim_high)
    plt.ylim(lim_low, lim_high)
    plt.grid()
    plt.xlabel('Reference Glucometer')
    plt.ylabel('Estimated Glucose')
    plt.title('Predicted results on Clark Grid')
    plt.show()




