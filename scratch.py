import vna_proc_utils
#vna_proc_utils.random_prediction(500, [60, 260])
#vna_proc_utils.geris_library_prediction('test_data_validated.csv', [80, 180], 21700)
#exit()

option = 4
id1 = 3
id2 = 10
id3 = 13

# Force the same frequency selection (10 frequencies), but not the same coefficient values for all subjects
if option == 0:
    vna_proc_utils.feature_extract(filename='test_data_validated.csv', measurement_range=[60, 200], valid_only=1,
                                   lasso_Alpha=None, subjects_id=[id1, id2, id3])
# Force only the same number of features but not the same frequencies
if option == 1:
    vna_proc_utils.feature_extract(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, lasso_Alpha = None, subjects_id = [id1]) #Dia 3, Assaf 5, Yaarit 6, Alan 13
    vna_proc_utils.feature_extract(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, lasso_Alpha = None, subjects_id = [id2])
    vna_proc_utils.feature_extract(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, lasso_Alpha = None, subjects_id = [id3])

# Force the use of one frequency only
if option == 2:
    vna_proc_utils.single_freq_predict(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, single_frequency_indx = None, subjects_id = [id1])
    vna_proc_utils.single_freq_predict(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, single_frequency_indx = None, subjects_id = [id2])
    vna_proc_utils.single_freq_predict(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, single_frequency_indx = None, subjects_id = [id3])

# Use a combination of the resonance frequency and the Q
if option == 3:
    vna_proc_utils.resonance_predict(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, subjects_id = [id1], method = 'Lasso') # 'SVR' (default) or 'Lasso'
    vna_proc_utils.resonance_predict(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, subjects_id = [id2], method = 'Lasso')
    vna_proc_utils.resonance_predict(filename = 'test_data_validated.csv', measurement_range=[60, 200], valid_only=1, subjects_id = [id3], method = 'Lasso')

if option == 4:
    vna_proc_utils.lasso_performance_on_noise(50, 400, [70, 210], type='lasso', target_num_freq = 10)
    vna_proc_utils.lasso_performance_on_noise(50, 400, [70, 210], type='lasso', target_num_freq = 10)
    vna_proc_utils.lasso_performance_on_noise(50, 400, [70, 210], type='lasso', target_num_freq = 10)