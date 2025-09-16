import numpy as np
import json
import vna_proc_utils
import csv
import matplotlib.pyplot as plt
import re

# Check if a db result complies with a given db filter
def check_filter(res, required_filter, sufficient_filter, exclude_filter):
    # First check that the glucose recorded is within [min_gluk, max_gluk], if not, remove result
    min_gluk = 30.0
    max_gluk = 280.0
    center_gluk = (max_gluk + min_gluk) / 2.0
    span_gluk = max_gluk - min_gluk
    try: # failing to read gluk level means the number was not inserted properly- remove this result
        g1 = int(res['gluk_level1'])
        g2 = int(res['gluk_level2'])
        g3 = int(res['gluk_level3'])
        if (abs(g1 - center_gluk) > span_gluk / 2.0) or (
                abs(g2 - center_gluk) > span_gluk / 2.0) or (
                abs(g3 - center_gluk) > span_gluk / 2.0):
            return 0
    except:
        return 0

    keysList = list(exclude_filter.keys())
    if keysList: # exclude filter not empty
        ok = 0
        for key in keysList: # if all exclude conditions are met then reject
            if not res[key] in exclude_filter[key]:
                ok = 1 # condition not met- accept
                break
        if not ok:
            return 0
    keysList = list(required_filter.keys())
    for key in keysList: # if one required condition not met then reject
        if not res[key] in required_filter[key]:
            return 0
    keysList = list(sufficient_filter.keys())
    if not keysList: # if condition is empty accept
        return 1
    else:
        for key in keysList: # if at least one condition met accept
            if res[key] in sufficient_filter[key]:
                return 1
    return 0

def patient2id(patient_id, patients):
    id = 1
    for p in patients:
        if p['patient_id'] == patient_id:
            return id
        id += 1
    return None # paitient not found

def extract_integer(s):
    match = re.search(r'-?\d+', s)
    if match:
        return int(match.group())
    else:
        return None  # or raise ValueError("No integer found")

def gluk_level_number(s1, s2, s3):
    k = 0
    s1 = extract_integer(s1)
    if s1:
        s1 = int(s1)
        k += 1
    else:
        s1 = 0
    s2 = extract_integer(s2)
    if s2:
        s2 = int(s2)
        k += 1
    else:
        s2 = 0
    s3 = extract_integer(s3)
    if s3:
        s3 = int(s3)
        k += 1
    else:
        s3 = 0
    return(s1+s2+s3)/k

def generate_data(gluk_consideration_range, results, required_filter, sufficient_filter, exclude_filter, patients):
    scan_list = []
    gluk_list = []
    id_list = []
    time_list = []

    for res in results:
        ok = check_filter(res, required_filter, sufficient_filter, exclude_filter)
        if ok == 1:
            gluk = np.round(gluk_level_number(res['gluk_level1'], res['gluk_level2'], res['gluk_level3']))
            if gluk < gluk_consideration_range[0] or gluk > gluk_consideration_range[1]:
                continue
            f = res['FREQ']
            s = res['scan_data'][-1]
            if {'$numberDouble': '-Infinity'} in s:
                continue
            id = patient2id(res['patient_id'], patients)
            id_list.append(id)
            scan_list.append(s)
            gluk_list.append(gluk)
            tstamp = [res['time'][:2]+':'+res['time'][2:4]+':'+res['time'][4:6]+'-'+ res['time'][-6:-4] + '-'+res['time'][-4:-2]+'-'+res['time'][-2:]]
            time_list.append(tstamp)

    import pandas as pd
    X = np.array(scan_list)
    Y = pd.DataFrame(X).T
    Y.columns = gluk_list
    Y.index = f
    Y.to_csv('rawdata.csv', index=True)
    plt.show()
    num_points = 400
    freq = np.linspace(f[0], f[-1], num_points)
    measurements = []
    k = 0
    for s in scan_list:
        measurements.append(np.interp(freq, f, s))
    return {'SUBJECT_ID': id_list, 'GLUCOSE': gluk_list, 'MEASUREMENT': measurements, 'TIME': time_list}

if  __name__ == '__main__':
    fc = 22000
    fs = 8000
    gluk_consideration_range = [50, 250]
    ANT_ID = '30' # Albert antennas tested: 13, 14, 15, 16 (90deg), 17, 18, 20, 21, 22, 23, 24, 25 (90 deg), 26, 27, 28, 29
    vf_vector = []
    power = -10
    db_antennas = vna_proc_utils.db_antennas
    db_scans = vna_proc_utils.db_scans
    db_patients = vna_proc_utils.db_patients
    with open(db_scans, 'r') as input_file1:
        results = json.load(input_file1)
    with open(db_patients, 'r') as input_file2:
        patients = json.load(input_file2)
    with open(db_antennas, 'r') as input_file3:
        antennas = json.load(input_file3)
    ant_names = [x['antena_name'] for x in antennas]
    ant_ids = [x['antena_id'] for x in antennas]
    i = ant_ids.index(ANT_ID)
    ant_name = ant_names[i]
    #required_filter = {'ANT_ID': [ANT_ID], 'date': ['240725'], 'patient_id': ['060471380', '038189916', '029971504']} # all fields must match # Assaf: 060471380 Dia: 038189916, Yaarit: 029971504, Dafna: 206459992, Alen: 0547269362
    #required_filter = {'ANT_ID': [ANT_ID], 'date': ['010925']} # Use this for time-domain test with Hanna on Sep-1-2025
    required_filter = {'ANT_ID': [ANT_ID]}
    sufficient_filter = {} # if not empty, one field must match to force inclusion
    #exclude_filter = {'patient_id': ['038189916'], 'date': ['020225']}
    exclude_filter = {'date': ['010925']} # use this for antenna 22 3-patient test
    #exclude_filter = {}
    results = generate_data(gluk_consideration_range, results, required_filter, sufficient_filter, exclude_filter, patients)
    header = ['SUBJECT_ID', 'GLUCOSE', 'MEASUREMENT','TIME']
    subject_ids = set(results['SUBJECT_ID'])
    print(f'Subjects IDs: {subject_ids}')
    #t = [x[0] for x in results['TIME']]
    #t_str = [str(datetime.strptime(x, '%H:%M:%S-%d-%m-%y').date()) for x in t]
    rows = zip(results['SUBJECT_ID'], results['GLUCOSE'], results['MEASUREMENT'], results['TIME'])
    filename = 'test_data.csv'
    with open(filename, 'w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(header)
        writer.writerows(rows)

    title_str = f'Antenna {ant_name}'
    valid_column = vna_proc_utils.add_additional_data(filename, do_plot = 1, title_str = title_str)
    valid_column = [int(x == True) for x in valid_column]
    header += ['VALID']
    rows = zip(results['SUBJECT_ID'], results['GLUCOSE'], results['MEASUREMENT'], results['TIME'], valid_column)
    filename = 'test_data_validated.csv'
    with open(filename, 'w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(header)
        writer.writerows(rows)
    exit(0)
