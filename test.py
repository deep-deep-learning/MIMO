import xlwt
from xlwt import Workbook

import numpy as np

import matplotlib.pyplot as plt

'''
wb = Workbook()

encoder_Si = {'fc1': np.load('encoder_Si_fc1.npy'), 'fc2': np.load('encoder_Si_fc2.npy'), 'fc3': np.load('encoder_Si_fc3.npy')}

sheet1 = wb.add_sheet('Sheet 1')
for i in range(encoder_Si['fc1'].shape[0]):
    sheet1.write(i,0,encoder_Si['fc1'][i])

sheet2 = wb.add_sheet('Sheet 2')
for i in range(encoder_Si['fc2'].shape[0]):
    sheet2.write(i,0,encoder_Si['fc2'][i])

sheet3 = wb.add_sheet('Sheet 3')
for i in range(encoder_Si['fc3'].shape[0]):
    sheet3.write(i,0,encoder_Si['fc3'][i])

wb.save('encoder_Si.xls')
'''

SER_full_model = np.load('First Try/SER_full_model.npy')
SER_reduced_model = np.load('First Try/SER_reduced_model.npy')

plt.figure(figsize=(16,9))

pl1, pl2 = plt.semilogy(np.arange(0,26,2), SER_full_model, '-*',
                           np.arange(0,26,2), SER_reduced_model, '-<')
# , SNR_array, SER, '-o'
plt.grid()
plt.legend([pl1, pl2],['Full', 'Reduced'])
plt.savefig('SER.png')