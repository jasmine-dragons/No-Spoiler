import matplotlib.pyplot as plt
import numpy as np

AUC_data = {
    'distil_bert_tvtropes': 0.85783,
    'distil_bert_goodreads': 0.82861,
    'lstm_tvtropes': 0.5963384813384813, #0.9194790705369822,
    'lstm_goodreads': 0.4999877020539233,
}

a = np.array([0, 0.5])
w = 0.16

fig,ax = plt.subplots()
ax.bar(a-w/2,[0.4999877020539233, 0.5963384813384813],w,color='#d52b93',label="LSTM")
ax.bar(a+w/2,[0.82861, 0.85783],w,color='#49006a',label="DistilBERT")
ax.set_xticks(a)
ax.set_xticklabels(('Goodreads', 'TV Tropes'))
ax.set_title('Comparison of AUC scores across models and training sets')
plt.xlabel('Training dataset')
plt.ylabel('AUC score')
plt.ylim(0, 1.125)
plt.legend()
plt.show()