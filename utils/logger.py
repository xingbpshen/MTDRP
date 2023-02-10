import pandas as pd
import torch


class Logger:
    # keys should be a 1D list. e.g ['epoch', 'name1', 'name2"]
    # we always set the first element (epoch) as the index
    def __init__(self, names, index=0):
        self.df = pd.DataFrame(columns=names)
        # self.df = self.df.set_index(names[index])
        self.index = index
        self.names = names
        self.col_num = len(names)

    # dictionary e.g {'epoch': 1, 'name1': 1.0, 'name2': 2.0}
    def log(self, dictionary):
        row = []
        for key in self.names:
            if type(dictionary[key]) == torch.Tensor:
                row.append(float(dictionary[key]))
            else:
                row.append(dictionary[key])
        # if df is empty
        if len(self.df) == 0:
            self.df.loc[0] = row
        else:
            self.df.loc[self.df.index.max() + 1] = row

    def save_plot(self, path, option='all'):
        if option == 'all':
            tmp_df = self.df.set_index(self.names[self.index])
            plot = tmp_df.plot()
            fig = plot.get_figure()
            fig.savefig(path)

    def save_csv(self, path, option='all'):
        if option == 'all':
            self.df.to_csv(path)
